#include "i2c_helpers.h"
#include "InitFusion.h"
#include "Fusion.h"
#include "ism330dlc.h"
#include "tusb.h"
#include "cdc_console.h"
#include "status_led.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "settings.h"
#include "output.h"
#include <math.h>
#include <pico/multicore.h>
#include <pico/stdlib.h>
#include <string.h>

// MAX_SENSORS and FLOATS_PER_SENSOR defined in output.h

typedef struct imu_stream_slot_t {
    uint32_t sequence;
    uint32_t timestamp_us;
    uint8_t sensor_count;
    float sensors_data[MAX_SENSORS][FLOATS_PER_SENSOR];
} imu_stream_slot_t;

typedef struct usb_stream_bridge_t {
    volatile bool descriptor_ready;
    volatile bool stream_started;
    volatile uint16_t sample_rate_hz;
    volatile uint8_t sensor_count;
    uint8_t bus_ids[MAX_SENSORS];
    uint8_t sensor_addrs[MAX_SENSORS];
    imu_stream_slot_t slots[2];
    volatile uint32_t published_sequence;
    volatile uint8_t published_slot;
} usb_stream_bridge_t;

static usb_stream_bridge_t g_usb_stream = {0};

typedef struct sensor_runtime_t {
    float period_ms;
    uint8_t sensor_count;
    Sensor sensors[MAX_SENSORS];
} sensor_runtime_t;

static sensor_runtime_t g_sensor_runtime = {0};

static void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors, uint8_t sensor_count);

static inline FusionQuaternion enforce_quaternion_continuity(FusionQuaternion q, FusionQuaternion q_ref) {
    // Choose hemisphere (q or -q) closest to reference quaternion
    float dot = q.element.w * q_ref.element.w +
                q.element.x * q_ref.element.x +
                q.element.y * q_ref.element.y +
                q.element.z * q_ref.element.z;

    if (dot < 0.0f) {
        q.element.w = -q.element.w;
        q.element.x = -q.element.x;
        q.element.y = -q.element.y;
        q.element.z = -q.element.z;
    }
    return q;
}

static inline float tilt_pitch_deg_from_gravity(const FusionVector gravity) {
    const float gy = gravity.axis.y;
    const float gz = gravity.axis.z;
    return (180.0f / (float)M_PI) * atan2f(-gravity.axis.x, sqrtf(gy * gy + gz * gz));
}

static inline float tilt_roll_deg_from_gravity(const FusionVector gravity) {
    return (180.0f / (float)M_PI) * atan2f(gravity.axis.y, gravity.axis.z);
}

static void stream_publish_descriptor(uint16_t sample_rate_hz, uint8_t sensor_count,
                                      const uint8_t *sensor_bus_ids, const uint8_t *sensor_addrs) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    memcpy((void *)g_usb_stream.bus_ids, sensor_bus_ids, sensor_count);
    memcpy((void *)g_usb_stream.sensor_addrs, sensor_addrs, sensor_count);
    g_usb_stream.sample_rate_hz = sample_rate_hz;
    g_usb_stream.sensor_count = sensor_count;
    __atomic_store_n(&g_usb_stream.descriptor_ready, true, __ATOMIC_RELEASE);
}

static void stream_publish_frame(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR], uint8_t sensor_count) {
    if (sensor_count > MAX_SENSORS) {
        sensor_count = MAX_SENSORS;
    }

    const uint8_t next_slot = (uint8_t)((__atomic_load_n(&g_usb_stream.published_slot, __ATOMIC_RELAXED) + 1u) & 0x01u);
    imu_stream_slot_t *slot = &g_usb_stream.slots[next_slot];
    slot->timestamp_us = (uint32_t)ts_us;
    slot->sensor_count = sensor_count;
    memcpy(slot->sensors_data, sensors_data, sizeof(slot->sensors_data));

    const uint32_t next_sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_RELAXED) + 1u;
    slot->sequence = next_sequence;
    __atomic_store_n(&g_usb_stream.published_slot, next_slot, __ATOMIC_RELEASE);
    __atomic_store_n(&g_usb_stream.published_sequence, next_sequence, __ATOMIC_RELEASE);
    __atomic_store_n(&g_usb_stream.stream_started, true, __ATOMIC_RELEASE);
}

static bool stream_copy_latest(imu_stream_slot_t *out, uint32_t *last_sequence) {
    while (true) {
        const uint32_t sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_ACQUIRE);
        if ((sequence == 0u) || (sequence == *last_sequence)) {
            return false;
        }

        const uint8_t slot_index = __atomic_load_n(&g_usb_stream.published_slot, __ATOMIC_ACQUIRE);
        memcpy(out, &g_usb_stream.slots[slot_index], sizeof(*out));

        const uint32_t confirm_sequence = __atomic_load_n(&g_usb_stream.published_sequence, __ATOMIC_ACQUIRE);
        if ((confirm_sequence == sequence) && (out->sequence == sequence)) {
            *last_sequence = sequence;
            return true;
        }
    }
}

static void sleep_until_deadline(uint64_t deadline_us) {
    while (true) {
        const uint64_t now_us = time_us_64();
        if (now_us >= deadline_us) {
            return;
        }

        const uint64_t remaining_us = deadline_us - now_us;
        if (remaining_us > 1000u) {
            sleep_us(500);
        } else if (remaining_us > 100u) {
            sleep_us((uint32_t)(remaining_us - 50u));
        } else {
            tight_loop_contents();
        }
    }
}

static void usb_transport_loop(void) {
    uint32_t last_sequence = 0;
    uint8_t descriptor_retries_remaining = 5;
    uint64_t next_descriptor_us = time_us_64();

    while (true) {
        tud_task();

        const uint64_t now_us = time_us_64();
        if (__atomic_load_n(&g_usb_stream.descriptor_ready, __ATOMIC_ACQUIRE) &&
            (descriptor_retries_remaining > 0u) &&
            (now_us >= next_descriptor_us)) {
            send_descriptor_frame(
                g_usb_stream.sample_rate_hz,
                g_usb_stream.sensor_count,
                g_usb_stream.bus_ids,
                g_usb_stream.sensor_addrs);
            descriptor_retries_remaining--;
            next_descriptor_us = now_us + 200000u;
        }

        imu_stream_slot_t latest;
        if (stream_copy_latest(&latest, &last_sequence)) {
            print_output_data(latest.timestamp_us, latest.sensors_data, latest.sensor_count);
            continue;
        }

        if (!__atomic_load_n(&g_usb_stream.stream_started, __ATOMIC_ACQUIRE)) {
            sleep_ms(1);
        } else {
            sleep_us(500);
        }
    }
}

static void sensor_core1_main(void) {
    float sensors_data[MAX_SENSORS][FLOATS_PER_SENSOR] = {{0.0f}};
    update_loop(g_sensor_runtime.period_ms, sensors_data, g_sensor_runtime.sensors, g_sensor_runtime.sensor_count);
}

// Main sensor loop - clean AHRS, no software filtering
static void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors, uint8_t sensor_count) {
    const uint64_t target_us = (uint64_t)(period_ms * 1000.0f + 0.5f);
    uint64_t next_loop_us = time_us_64();

    while (true) {
        read_all_sensors(sensors);

        for (uint8_t i = 0; i < sensor_count; i++) {
            // Apply calibration (identity by default)
            sensors[i].gyroscope = FusionCalibrationInertial(
                sensors[i].gyroscope,
                sensors[i].calibration.gyroscopeMisalignment,
                sensors[i].calibration.gyroscopeSensitivity,
                sensors[i].calibration.gyroscopeOffset);

            sensors[i].accelerometer = FusionCalibrationInertial(
                sensors[i].accelerometer,
                sensors[i].calibration.accelerometerMisalignment,
                sensors[i].calibration.accelerometerSensitivity,
                sensors[i].calibration.accelerometerOffset);

            // Gyro offset tracking (stationary bias removal)
            sensors[i].gyroscope = FusionOffsetUpdate(&sensors[i].offset, sensors[i].gyroscope);

            // Compute delta time
            const float deltaTime = (float)(sensors[i].timestamp - sensors[i].previousTimestamp) / 1e6f;
            sensors[i].previousTimestamp = sensors[i].timestamp;

            // AHRS update (no magnetometer, heading fixed at 0)
            FusionAhrsUpdateExternalHeading(&sensors[i].ahrs, sensors[i].gyroscope, sensors[i].accelerometer, 0.0f, deltaTime);

            // Get quaternion and enforce continuity
            FusionQuaternion quat = FusionAhrsGetQuaternion(&sensors[i].ahrs);
            quat = enforce_quaternion_continuity(quat, sensors[i].previousQuaternion);
            sensors[i].previousQuaternion = quat;

            // Derive tilt from gravity so pitch/roll stay stable under yaw motion
            const FusionVector gravity = FusionAhrsGetGravity(&sensors[i].ahrs);
            sensors[i].pitchDeg = tilt_pitch_deg_from_gravity(gravity);
            sensors[i].rollDeg = tilt_roll_deg_from_gravity(gravity);

            // Output: quaternion + gyro
            sensors_data[i][0] = quat.element.w;
            sensors_data[i][1] = quat.element.x;
            sensors_data[i][2] = quat.element.y;
            sensors_data[i][3] = quat.element.z;
            sensors_data[i][4] = sensors[i].gyroscope.axis.x;
            sensors_data[i][5] = sensors[i].gyroscope.axis.y;
            sensors_data[i][6] = sensors[i].gyroscope.axis.z;
        }

        uint64_t out_ts = time_us_64();
        stream_publish_frame(out_ts, sensors_data, sensor_count);

        // Maintain a stable absolute deadline instead of sleeping relative to
        // the loop start each iteration.
        next_loop_us += target_us;
        uint64_t schedule_now_us = time_us_64();

        if (schedule_now_us < next_loop_us) {
            sleep_until_deadline(next_loop_us);
        } else if ((schedule_now_us - next_loop_us) > target_us) {
            // If we fall badly behind, re-anchor to the current time rather
            // than carrying the slip forever.
            next_loop_us = schedule_now_us;
        }
    }
}

int main() {
    // Init TinyUSB directly — do NOT call stdio_init_all() as the
    // stdio USB driver steals bytes from our tud_cdc_read() calls.
    tusb_init();

    // Init NeoPixel status LED first (no USB needed)
    status_led_init();
    status_led_set(STATUS_BOOT);  // Amber — waiting for USB

    // Setup I2C buses
    if (!setup_I2C_pins()) {
        status_led_set(STATUS_ERROR);
        send_control_msg(MSG_TYPE_ERR_I2C);
        while (1) { tud_task(); sleep_ms(100); }
    }

    while (!tud_cdc_connected()) {
        status_led_set(STATUS_BOOT);
        tud_task();
        sleep_ms(10);
    }

    status_led_set(STATUS_CFG_WAIT);
    wait_for_settings();

    status_led_set(STATUS_INIT);

    if (!initialize_sensors()) {
        status_led_set(STATUS_ERROR);
        send_control_msg(MSG_TYPE_ERR_IMU);
        while (1) { tud_task(); sleep_ms(100); }
    }

    uint8_t sensor_count = get_active_sensor_count();
    if (sensor_count > MAX_SENSORS) sensor_count = MAX_SENSORS;
    uint8_t sensor_bus_ids[MAX_SENSORS];
    uint8_t sensor_addrs[MAX_SENSORS];
    for (uint8_t i = 0; i < sensor_count; i++) {
        sensor_bus_ids[i] = get_active_sensor_bus(i);
        sensor_addrs[i] = get_active_sensor_addr(i);
    }

    g_sensor_runtime.sensor_count = sensor_count;
    g_sensor_runtime.period_ms = 1000.0f / (float)imu_reader_settings.sampleRate;
    initialize_sensors_values(g_sensor_runtime.sensors, MAX_SENSORS);
    initialize_calibrations(g_sensor_runtime.sensors, MAX_SENSORS);
    initialize_algos(g_sensor_runtime.sensors, MAX_SENSORS);

    stream_publish_descriptor((uint16_t)imu_reader_settings.sampleRate, sensor_count, sensor_bus_ids, sensor_addrs);
    status_led_set(STATUS_STREAM);
    multicore_launch_core1(sensor_core1_main);
    usb_transport_loop();

    return 0;
}
