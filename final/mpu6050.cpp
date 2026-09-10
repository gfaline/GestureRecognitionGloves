#include "mpu6050.h"
#include <Wire.h>

// ------------------------------------------------------------
// Low-level register helpers
// ------------------------------------------------------------
static bool wireStarted = false;

static void ensureWire() {
  if (!wireStarted) {
    Wire.begin();
    Wire.setClock(400000);  // 400 kHz fast mode
    wireStarted = true;
  }
}

static void writeReg(uint8_t addr, uint8_t reg, uint8_t value) {
  ensureWire();
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Read `count` bytes starting at `reg` into `buf`. Returns bytes read.
static uint8_t readRegs(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t count) {
  ensureWire();
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0;  // repeated start

  uint8_t got = Wire.requestFrom((int)addr, (int)count, (int)true);
  for (uint8_t i = 0; i < got && Wire.available(); i++) {
    buf[i] = Wire.read();
  }
  return got;
}

// Combine two bytes (high, low) into a signed 16-bit value.
static int16_t combine(uint8_t hi, uint8_t lo) {
  return (int16_t)((hi << 8) | lo);
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
void wakeSensor(uint8_t addr) {
  static bool configured = false;

  // Clear SLEEP bit, use gyro X PLL as clock source (0x01).
  writeReg(addr, MPU_REG_PWR_MGMT_1, 0x01);

  if (!configured) {
    // DLPF ~44 Hz accel / 42 Hz gyro (CONFIG = 3)
    writeReg(addr, MPU_REG_CONFIG, 0x03);
    // Sample rate divider: 1 kHz / (1 + 7) = 125 Hz
    writeReg(addr, MPU_REG_SMPLRT_DIV, 0x07);
    // Gyro full scale +/- 250 dps
    writeReg(addr, MPU_REG_GYRO_CONFIG, 0x00);
    // Accel full scale +/- 2 g
    writeReg(addr, MPU_REG_ACCEL_CONFIG, 0x00);
    configured = true;
  }
}

void readGyroData(uint8_t addr, float &gx, float &gy, float &gz) {
  uint8_t b[6];
  if (readRegs(addr, MPU_REG_GYRO_XOUT_H, b, 6) < 6) {
    gx = gy = gz = 0.0f;
    return;
  }
  gx = (float)combine(b[0], b[1]);
  gy = (float)combine(b[2], b[3]);
  gz = (float)combine(b[4], b[5]);
}

void rawGyroToDPS(float rawGX, float rawGY, float rawGZ,
                  float &dpsGX, float &dpsGY, float &dpsGZ) {
  dpsGX = rawGX / MPU_GYRO_LSB_PER_DPS;
  dpsGY = rawGY / MPU_GYRO_LSB_PER_DPS;
  dpsGZ = rawGZ / MPU_GYRO_LSB_PER_DPS;
}

void readAccelData(uint8_t addr, float &ax, float &ay, float &az) {
  uint8_t b[6];
  if (readRegs(addr, MPU_REG_ACCEL_XOUT_H, b, 6) < 6) {
    ax = ay = az = 0.0f;
    return;
  }
  ax = (float)combine(b[0], b[1]);
  ay = (float)combine(b[2], b[3]);
  az = (float)combine(b[4], b[5]);
}

void rawAccelToGForce(float rawAX, float rawAY, float rawAZ,
                      float &gForceAX, float &gForceAY, float &gForceAZ) {
  gForceAX = rawAX / MPU_ACCEL_LSB_PER_G;
  gForceAY = rawAY / MPU_ACCEL_LSB_PER_G;
  gForceAZ = rawAZ / MPU_ACCEL_LSB_PER_G;
}
