#ifndef ISM330DHCL_REGISTERS_H
#define ISM330DHCL_REGISTERS_H
// ISM330DHCX register addresses (ST DS13012)
#define WHO_AM_I 0x0F
#define CTRL1_XL 0x10  // Accelerometer ODR / full scale / LPF2 enable (bit 1)
#define CTRL2_G 0x11   // Gyroscope ODR / full scale
#define CTRL3_C 0x12   // BDU, IF_INC, SW_RESET (bit 0), BOOT (bit 7)
#define CTRL4_C 0x13   // LPF1_SEL_G (bit 1) — routes the gyro LPF1 into the path
#define CTRL6_C 0x15   // Gyro LPF1 bandwidth, FTYPE[2:0]
#define CTRL7_G 0x16   // Gyro advanced settings (not used here)
#define CTRL8_XL 0x17  // Accel LPF2/HP cutoff, HPCF_XL[2:0] in bits [7:5]
#define CTRL9_XL 0x18  // DEVICE_CONF (bit 1) — ST-recommended startup bit
#define STATUS_REG 0x1E
#define OUTX_L_G 0x22  // Gyroscope output registers
#define OUTX_L_XL 0x28 // Accelerometer output registers
// ISM330DHCX I2C address (SDO/SA0 pin low)
#define ISM330DHCX_ADDR_DO_LOW 0x6A
// ISM330DHCX I2C address (SDO/SA0 pin high)
#define ISM330DHCX_ADDR_DO_HIGH 0x6B

// Use 0x6B if SDO/SA0 pin is high
// Expected WHO_AM_I value for ISM330DHCX
#define ISM330DHCX_ID 0x6B
#endif
