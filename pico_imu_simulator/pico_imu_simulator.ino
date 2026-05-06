// pico_imu_simulator.ino
// Simulates 4 IMUs over USB serial using the same binary protocol as pico_imu_reader.
// Target: Seeed XIAO RP2040 (Arduino IDE, board: "Seeed XIAO RP2040")
//
// Sensor layout matches control_config.yaml imu_mapping:
//   [0] base   — slew/yaw dominant (±45° Z), no pitch, small roll noise
//   [1] boom   — pitch oscillation 0.15 Hz
//   [2] bucket — pitch oscillation 0.12 Hz
//   [3] arm    — pitch oscillation 0.18 Hz
// All sensors share the same yaw plane. Roll adds a small slow oscillation on all.

#include <math.h>

// ---- Protocol constants (must match pico_imu_reader / usb_serial_reader.py) ----
#define FRAME_SYNC_0           0xAA
#define FRAME_SYNC_1           0x55
#define FRAME_VERSION_DATA     1
#define FRAME_VERSION_CTRL     2
#define FRAME_VERSION_DESC     3
#define FRAME_VERSION_CAL_REPORT 5

#define MSG_TYPE_CFG_OK        0x01
#define MSG_TYPE_CFG_WAIT      0x02
#define MSG_TYPE_ERROR         0x04
#define MSG_TYPE_ERR_I2C       0x05
#define MSG_TYPE_ERR_IMU       0x06
#define MSG_TYPE_CAL_WAIT      0x07
#define MSG_TYPE_ERR_CAL       0x08

#define CAL_REPORT_ACCEPTED    0x01

#define NUM_SENSORS        4
#define FLOATS_PER_SENSOR  7   // w, x, y, z, gx, gy, gz

// Simulated calibration duration in milliseconds (keep short for fast iteration)
#define SIM_CAL_DURATION_MS 3000

// ---- Settings (populated from host config string) ----
static int   sampleRate       = 200;
static float gyroRangeDps     = 250.0f;
static float ahrsGain         = 5.0f;
static float accelRejectionDeg = 20.0f;
static float recoveryS        = 0.5f;

// ---- Simulation state ----
static uint32_t loopPeriodUs;
static float    simTime = 0.0f;

// Per-IMU pitch: [0]=base has no pitch, [1..3]=arm links oscillate ±45°
static const float pitchAmpDeg[NUM_SENSORS]   = {  0.0f, 45.0f, 45.0f, 45.0f };
static const float pitchFreqHz[NUM_SENSORS]   = {  0.0f,  0.15f, 0.12f, 0.18f };
static const float pitchPhaseRad[NUM_SENSORS]  = {  0.0f,  0.0f,  1.0f,  2.0f };

// Per-IMU roll: small slow oscillation on all sensors
static const float rollAmpDeg[NUM_SENSORS]    = {  2.0f,  3.0f,  3.0f,  3.0f };
static const float rollFreqHz[NUM_SENSORS]    = {  0.08f, 0.07f, 0.09f, 0.06f };
static const float rollPhaseRad[NUM_SENSORS]  = {  0.0f,  2.1f,  0.8f,  3.5f };

// Yaw: single shared oscillation (same plane for all)
static const float yawFreqHz = 0.10f;

// ---- Descriptor info (I2C addresses matching real firmware layout) ----
// [0] base=bus1:0x6B  [1] boom=bus0:0x6A  [2] bucket=bus0:0x6B  [3] arm=bus1:0x6A
static const uint8_t imuBus[NUM_SENSORS]  = { 1, 0, 0, 1 };
static const uint8_t imuAddr[NUM_SENSORS] = { 0x6B, 0x6A, 0x6B, 0x6A };

// ---- Settings buffer ----
#define SETTINGS_BUF_LEN 256
static char settingsBuf[SETTINGS_BUF_LEN];
static int  settingsIdx = 0;

// ---- Helpers ----

static uint16_t calcChecksum(const uint8_t* buf, size_t start, size_t end) {
  uint16_t cs = 0;
  for (size_t i = start; i < end; i++) {
    cs += buf[i];
  }
  return cs;
}

static float gyroNoise(float sigma) {
  float sum = 0.0f;
  for (int i = 0; i < 6; i++) {
    sum += (float)random(-1000, 1001) / 1000.0f;
  }
  return sum * sigma / 3.0f;
}

// ---- Send control frame (version=2) ----
static void sendControlMsg(uint8_t msgType) {
  uint8_t frame[6];
  frame[0] = FRAME_SYNC_0;
  frame[1] = FRAME_SYNC_1;
  frame[2] = FRAME_VERSION_CTRL;
  frame[3] = msgType;
  uint16_t cs = frame[2] + frame[3];
  frame[4] = (uint8_t)(cs & 0xFF);
  frame[5] = (uint8_t)((cs >> 8) & 0xFF);
  Serial.write(frame, 6);
  Serial.flush();
}

// ---- Send descriptor frame (version=3) ----
static void sendDescriptor() {
  const size_t frameLen = 2 + 1 + 1 + 2 + NUM_SENSORS * 2 + 2;
  uint8_t frame[frameLen];
  size_t idx = 0;

  frame[idx++] = FRAME_SYNC_0;
  frame[idx++] = FRAME_SYNC_1;
  frame[idx++] = FRAME_VERSION_DESC;
  frame[idx++] = NUM_SENSORS;

  frame[idx++] = (uint8_t)(sampleRate & 0xFF);
  frame[idx++] = (uint8_t)((sampleRate >> 8) & 0xFF);

  for (int i = 0; i < NUM_SENSORS; i++) {
    frame[idx++] = imuBus[i];
    frame[idx++] = imuAddr[i];
  }

  uint16_t cs = calcChecksum(frame, 2, idx);
  frame[idx++] = (uint8_t)(cs & 0xFF);
  frame[idx++] = (uint8_t)((cs >> 8) & 0xFF);

  Serial.write(frame, idx);
  Serial.flush();
}

// ---- Send calibration report frame (version=5) ----
static void sendCalibrationReport() {
  // Frame: sync(2) + ver(1) + count(1) + duration_ms(4) + flags(1) + reserved(1)
  //        + thresholds(4*4=16) + N*(sample_count(4)+read_errors(4)+fail_flags(2)+reserved(2)+vals(15*4=60))
  //        + checksum(2)
  // = 28 + N*72
  const size_t perSensor = 72;
  const size_t frameLen = 28 + NUM_SENSORS * perSensor;
  uint8_t frame[frameLen];
  size_t idx = 0;

  frame[idx++] = FRAME_SYNC_0;
  frame[idx++] = FRAME_SYNC_1;
  frame[idx++] = FRAME_VERSION_CAL_REPORT;
  frame[idx++] = NUM_SENSORS;

  // duration_ms
  uint32_t dur = SIM_CAL_DURATION_MS;
  memcpy(&frame[idx], &dur, 4); idx += 4;

  // flags (accepted) + reserved
  frame[idx++] = CAL_REPORT_ACCEPTED;
  frame[idx++] = 0;

  // thresholds: gyro_std_limit, accel_std_limit, accel_norm_min, accel_norm_max
  float thresholds[4] = { 0.5f, 0.02f, 0.9f, 1.1f };
  memcpy(&frame[idx], thresholds, 16); idx += 16;

  // Per-sensor data
  for (int i = 0; i < NUM_SENSORS; i++) {
    uint32_t sampleCount = (uint32_t)(SIM_CAL_DURATION_MS * sampleRate / 1000);
    uint32_t readErrors  = 0;
    uint16_t failFlags   = 0;
    uint16_t reserved    = 0;
    memcpy(&frame[idx], &sampleCount, 4); idx += 4;
    memcpy(&frame[idx], &readErrors,  4); idx += 4;
    memcpy(&frame[idx], &failFlags,   2); idx += 2;
    memcpy(&frame[idx], &reserved,    2); idx += 2;

    // 15 floats: gyro_bias(3) gyro_std(3) accel_mean(3) accel_std(3)
    //            accel_norm_mean accel_norm_min accel_norm_max
    float vals[15] = {
      0.01f,  0.01f,  0.01f,   // gyro_bias_dps
      0.05f,  0.05f,  0.05f,   // gyro_std_dps
      0.0f,   0.0f,   1.0f,    // accel_mean_g (Z-up)
      0.001f, 0.001f, 0.001f,  // accel_std_g
      1.0f,                     // accel_norm_mean_g
      0.99f,                    // accel_norm_min_g
      1.01f,                    // accel_norm_max_g
    };
    memcpy(&frame[idx], vals, 60); idx += 60;
  }

  uint16_t cs = calcChecksum(frame, 2, idx);
  frame[idx++] = (uint8_t)(cs & 0xFF);
  frame[idx++] = (uint8_t)((cs >> 8) & 0xFF);

  Serial.write(frame, idx);
  Serial.flush();
}

// ---- Send data frame (version=1) ----
static void sendDataFrame(uint32_t timestampUs, float data[][FLOATS_PER_SENSOR]) {
  const size_t payloadLen = 4 + NUM_SENSORS * FLOATS_PER_SENSOR * 4;
  const size_t frameLen   = 2 + 1 + 1 + payloadLen + 2;
  uint8_t frame[frameLen];
  size_t idx = 0;

  frame[idx++] = FRAME_SYNC_0;
  frame[idx++] = FRAME_SYNC_1;
  frame[idx++] = FRAME_VERSION_DATA;
  frame[idx++] = NUM_SENSORS;

  memcpy(&frame[idx], &timestampUs, 4);
  idx += 4;

  for (int i = 0; i < NUM_SENSORS; i++) {
    for (int j = 0; j < FLOATS_PER_SENSOR; j++) {
      float v = data[i][j];
      memcpy(&frame[idx], &v, 4);
      idx += 4;
    }
  }

  uint16_t cs = calcChecksum(frame, 2, idx);
  frame[idx++] = (uint8_t)(cs & 0xFF);
  frame[idx++] = (uint8_t)((cs >> 8) & 0xFF);

  Serial.write(frame, idx);
  Serial.flush();
}

// ---- Parse config string from host ----
static bool parseConfig(const char* buf) {
  const char* p;

  p = strstr(buf, "SR=");
  if (!p) return false;
  sampleRate = atoi(p + 3);
  if (sampleRate < 10)   sampleRate = 200;
  if (sampleRate > 1000) sampleRate = 1000;

  p = strstr(buf, "GYRO_DPS=");
  if (p) gyroRangeDps = atof(p + 9);

  p = strstr(buf, "GAIN=");
  if (p) ahrsGain = atof(p + 5);

  p = strstr(buf, "ACC_REJ=");
  if (p) accelRejectionDeg = atof(p + 8);

  p = strstr(buf, "RECOV_S=");
  if (p) recoveryS = atof(p + 8);

  loopPeriodUs = 1000000UL / (uint32_t)sampleRate;
  return true;
}

// Euler to quaternion: q = q_yaw * q_pitch * q_roll  (intrinsic RPY)
static void eulerToQuat(float rollDeg, float pitchDeg, float yawDeg,
                        float* w, float* x, float* y, float* z) {
  float cr = cosf(rollDeg  * (float)M_PI / 360.0f);
  float sr = sinf(rollDeg  * (float)M_PI / 360.0f);
  float cp = cosf(pitchDeg * (float)M_PI / 360.0f);
  float sp = sinf(pitchDeg * (float)M_PI / 360.0f);
  float cy = cosf(yawDeg   * (float)M_PI / 360.0f);
  float sy = sinf(yawDeg   * (float)M_PI / 360.0f);
  *w =  cy*cp*cr + sy*sp*sr;
  *x =  cy*cp*sr - sy*sp*cr;
  *y =  cy*sp*cr + sy*cp*sr;
  *z =  sy*cp*cr - cy*sp*sr;
}

// ---- Wait for config, then simulate calibration, then start streaming ----
static void wait_for_config() {
  uint32_t lastBeat = millis();

  // Phase 1: wait for host config
  while (true) {
    uint32_t now = millis();
    if (now - lastBeat >= 200) {
      sendControlMsg(MSG_TYPE_CFG_WAIT);
      lastBeat = now;
    }

    while (Serial.available()) {
      char c = Serial.read();
      if (settingsIdx < SETTINGS_BUF_LEN - 1) {
        settingsBuf[settingsIdx++] = c;
        settingsBuf[settingsIdx] = '\0';
      }

      if (c == '\n' || settingsIdx >= SETTINGS_BUF_LEN - 1) {
        if (parseConfig(settingsBuf)) {
          // Send CFG_OK for ~500ms
          uint32_t t0 = millis();
          while (millis() - t0 < 500) {
            sendControlMsg(MSG_TYPE_CFG_OK);
            delay(100);
          }
          goto calibration;
        }
        settingsIdx = 0;
        settingsBuf[0] = '\0';
      }
    }

    delay(10);
  }

calibration:
  // Phase 2: simulate startup calibration (send CAL_WAIT heartbeats)
  {
    uint32_t calStart = millis();
    uint32_t lastCalBeat = millis();
    while (millis() - calStart < SIM_CAL_DURATION_MS) {
      uint32_t now = millis();
      if (now - lastCalBeat >= 200) {
        sendControlMsg(MSG_TYPE_CAL_WAIT);
        lastCalBeat = now;
      }
      delay(10);
    }
    // Send calibration report (accepted)
    sendCalibrationReport();
  }

  // Phase 3: send descriptor and begin streaming
  sendDescriptor();
}

// ---- Arduino setup ----
void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
  loopPeriodUs = 1000000UL / (uint32_t)sampleRate;
}

// ---- Arduino loop ----
void loop() {
  static bool configured = false;
  if (!configured) {
    wait_for_config();
    configured = true;
  }

  uint32_t loopStart = micros();

  float dt = 1.0f / (float)sampleRate;
  simTime += dt;

  float yawDeg  = 45.0f * sinf(2.0f * (float)M_PI * yawFreqHz * simTime);
  float yawRate = 45.0f * 2.0f * (float)M_PI * yawFreqHz
                  * cosf(2.0f * (float)M_PI * yawFreqHz * simTime);

  const float gyroNoiseSigma = 0.3f;

  float sensorData[NUM_SENSORS][FLOATS_PER_SENSOR];

  for (int i = 0; i < NUM_SENSORS; i++) {
    float pitchDeg = pitchAmpDeg[i] * sinf(2.0f * (float)M_PI * pitchFreqHz[i] * simTime + pitchPhaseRad[i]);
    float rollDeg  = rollAmpDeg[i]  * sinf(2.0f * (float)M_PI * rollFreqHz[i]  * simTime + rollPhaseRad[i]);

    float w, x, y, z;
    eulerToQuat(rollDeg, pitchDeg, yawDeg, &w, &x, &y, &z);

    float pitchRate = pitchAmpDeg[i] * 2.0f * (float)M_PI * pitchFreqHz[i]
                      * cosf(2.0f * (float)M_PI * pitchFreqHz[i] * simTime + pitchPhaseRad[i]);
    float rollRate  = rollAmpDeg[i]  * 2.0f * (float)M_PI * rollFreqHz[i]
                      * cosf(2.0f * (float)M_PI * rollFreqHz[i]  * simTime + rollPhaseRad[i]);

    float gx = rollRate  + gyroNoise(gyroNoiseSigma);
    float gy = pitchRate + gyroNoise(gyroNoiseSigma);
    float gz = yawRate   + gyroNoise(gyroNoiseSigma);

    sensorData[i][0] = w;
    sensorData[i][1] = x;
    sensorData[i][2] = y;
    sensorData[i][3] = z;
    sensorData[i][4] = gx;
    sensorData[i][5] = gy;
    sensorData[i][6] = gz;
  }

  uint32_t ts = micros();
  sendDataFrame(ts, sensorData);

  uint32_t elapsed = micros() - loopStart;
  if (elapsed < loopPeriodUs) {
    delayMicroseconds(loopPeriodUs - elapsed);
  }
}
