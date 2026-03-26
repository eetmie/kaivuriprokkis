#ifndef OUTPUT_H
#define OUTPUT_H

#include <inttypes.h>
#include "imu_reader.h"

#define MAX_SENSORS 4
#define FLOATS_PER_SENSOR 7

// Message types for control frames (version=2)
#define MSG_TYPE_CFG_OK    0x01
#define MSG_TYPE_CFG_WAIT  0x02
#define MSG_TYPE_ERROR     0x04
#define MSG_TYPE_ERR_I2C   0x05  // I2C bus init failed
#define MSG_TYPE_ERR_IMU   0x06  // No IMU detected

void print_output_data(uint64_t ts_us, float sensors_data[][FLOATS_PER_SENSOR], uint8_t sensor_count);
void send_control_msg(uint8_t msg_type);
void send_descriptor_frame(uint16_t sample_rate_hz, uint8_t sensor_count, const uint8_t *bus_ids, const uint8_t *device_addrs);

extern imu_reader_settings_t imu_reader_settings;

#endif
