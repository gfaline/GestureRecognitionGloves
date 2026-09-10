#ifndef MPU6050_H
#define MPU6050_H

#include <Arduino.h>

// ============================================================
// Minimal MPU-6050 wrapper (raw Wire register access)
//
// Matches the API used by final.ino:
//   wakeSensor(addr)
//   readGyroData(addr, gx, gy, gz)        -> raw int16 counts (as float)
//   rawGyroToDPS(gx, gy, gz, dx, dy, dz)  -> degrees / second
//   readAccelData(addr, ax, ay, az)       -> raw int16 counts (as float)
//   rawAccelToGForce(ax, ay, az, gx, gy, gz) -> g (9.81 m/s^2 units)
//
// Assumes the default full-scale ranges configured by wakeSensor():
//   gyro  = +/- 250 dps   -> 131.0 LSB per dps
//   accel = +/- 2 g       -> 16384.0 LSB per g
//
// Wire.begin() is called automatically on the first wakeSensor() call,
// so the sketch does not need to call it in setup().
// ============================================================

// Sensitivity constants (default ranges)
const float MPU_GYRO_LSB_PER_DPS  = 131.0f;
const float MPU_ACCEL_LSB_PER_G   = 16384.0f;

// Register map
const uint8_t MPU_REG_SMPLRT_DIV   = 0x19;
const uint8_t MPU_REG_CONFIG       = 0x1A;
const uint8_t MPU_REG_GYRO_CONFIG  = 0x1B;
const uint8_t MPU_REG_ACCEL_CONFIG = 0x1C;
const uint8_t MPU_REG_ACCEL_XOUT_H = 0x3B;
const uint8_t MPU_REG_GYRO_XOUT_H  = 0x43;
const uint8_t MPU_REG_PWR_MGMT_1   = 0x6B;
const uint8_t MPU_REG_WHO_AM_I     = 0x75;

// Wake the device out of sleep and apply default configuration.
// Safe to call repeatedly (e.g. every loop iteration).
void wakeSensor(uint8_t addr);

// Read raw gyro counts (signed int16, returned as float, no scaling).
void readGyroData(uint8_t addr, float &gx, float &gy, float &gz);

// Convert raw gyro counts to degrees per second.
void rawGyroToDPS(float rawGX, float rawGY, float rawGZ,
                  float &dpsGX, float &dpsGY, float &dpsGZ);

// Read raw accelerometer counts (signed int16, returned as float, no scaling).
void readAccelData(uint8_t addr, float &ax, float &ay, float &az);

// Convert raw accelerometer counts to g.
void rawAccelToGForce(float rawAX, float rawAY, float rawAZ,
                      float &gForceAX, float &gForceAY, float &gForceAZ);

#endif // MPU6050_H
