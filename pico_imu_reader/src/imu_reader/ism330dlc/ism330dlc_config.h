#ifndef ISM330DLC_CONFIG_H
#define ISM330DLC_CONFIG_H

#include <stdint.h>

// What full scales this part supports, and how each is encoded in its CTRL
// register. This header describes the sensor's capabilities only — the ranges
// actually requested live next to every other tunable in settings.c, so gyro
// and accel are configured the same way instead of one being a #define here and
// the other a runtime setting.
//
// ODR is deliberately absent: it is derived from the requested rate by
// select_odr_bits() in ism330dlc.c rather than being fixed per sensor.
//
// Both tables must stay sorted ascending by scale — selection walks them in
// order and takes the first entry at or above the requested full scale.

typedef struct imu_full_scale_t {
    float   scale;   // full-scale magnitude (dps for gyro, g for accel)
    uint8_t mask;    // CTRL register bits selecting that full scale
} imu_full_scale_t;

// CTRL2_G: FS_125 is bit 1, FS_G[1:0] is bits [3:2] (00=250, 01=500, 10=1000,
// 11=2000 dps).
#define ISM330_GYRO_FULL_SCALES { \
    {  125.0f, 0x02 },            \
    {  250.0f, 0x00 },            \
    {  500.0f, 0x04 },            \
    { 1000.0f, 0x08 },            \
    { 2000.0f, 0x0C },            \
}

// CTRL1_XL: FS_XL[1:0] is bits [3:2]. This encoding is not monotonic —
// 00=2 g, 01=16 g, 10=4 g, 11=8 g — so 16 g is 0x04 and 8 g is 0x0C.
#define ISM330_ACCEL_FULL_SCALES { \
    {  2.0f, 0x00 },               \
    {  4.0f, 0x08 },               \
    {  8.0f, 0x0C },               \
    { 16.0f, 0x04 },               \
}

#endif //ISM330DLC_CONFIG_H
