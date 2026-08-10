#ifndef IMU_READER_H
#define IMU_READER_H

typedef struct imu_reader_settings_t {
    int   sampleRate;
    // Requested sensor full scales. initialize_sensors() rewrites both to the
    // nearest range the part actually supports, so everything downstream (the
    // raw-count conversion, Fusion's gyroscopeRange, the descriptor frame the
    // host logs) reads one already-normalized value.
    float gyroRangeDps;
    float accelRangeG;
    float ahrsGain;
    float ahrsAccelRejection;
    float ahrsRecoveryPeriodS;
} imu_reader_settings_t;

// Global settings (defined in settings.c, consumed by InitFusion.c)
extern imu_reader_settings_t imu_reader_settings;

#endif
