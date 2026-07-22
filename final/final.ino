#include <mpu6050.h>

#define MPU_ADDRESS 0x68  // 0x69 if AD0 is high

// ---------------- IMU variables ----------------
float rawGX, rawGY, rawGZ;
float rawAX, rawAY, rawAZ;
float dpsGX, dpsGY, dpsGZ;
float gForceAX, gForceAY, gForceAZ;

// ---------------- Vibrator Motor ----------------
const int motorPin = A6;

// Non-blocking pulse state
bool          motorOn        = false;
unsigned long motorOnMs      = 0;   // when the current ON phase started
unsigned long motorOffMs     = 0;   // when the current OFF phase started
int           motorPulsesLeft = 0;  // how many ON/OFF cycles remain

const unsigned long MOTOR_ON_MS  = 150;  // vibration burst duration
const unsigned long MOTOR_OFF_MS = 100;  // gap between bursts

// ---------------- Flex sensor readings ----------------
int sensor;   // A0
int sensor1;  // A1
int sensor2;  // A2
int sensor3;  // A3
int sensor4;  // A4
int button = A8;

// EMA filtered values
float f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
const float ALPHA = 0.2;

// ---------------- Calibration + normalization ----------------
float flexMin[5]  = {0, 0, 0, 0, 0};
float flexMax[5]  = {1, 1, 1, 1, 1};
float flexNorm[5] = {0, 0, 0, 0, 0};

bool minCalibrated = false;
bool maxCalibrated = false;

// ---------------- Operating mode ----------------
// DETECT = gesture recognition; LOG = CSV output
enum Mode { MODE_DETECT, MODE_LOG };
Mode currentMode = MODE_DETECT;

// ---------------- Data logging (LOG mode) ----------------
int gestureID = 0;
bool softButton = false;         // toggled by 'b' key in Serial Monitor
unsigned long logStartMs    = 0;
unsigned long lastLogUs     = 0;
unsigned long sampleHz      = 50;
unsigned long samplePeriodUs = 20000;   // 1e6 / 50

// ---------------- Gesture detection (DETECT mode) ----------------
const unsigned long DWELL_MS  = 1250;
const unsigned long REARM_MS  = 200;

unsigned long holdStartMs    = 0;
unsigned long releaseStartMs = 0;
bool latched = false;

enum GestureID_t { G_NONE, G_OK, G_GUN, G_STOP, G_UP, G_DOWN, G_BLINK,
                   G_LOOK, G_PHOTO, G_COOL, G_CUT, G_ROCK, G_PROBLEM, G_POINT };
GestureID_t activeGesture = G_NONE;

// ---------------- Stop / Blink thresholds ----------------
// Stop  = fist held for 500 ms (single hold)
// Blink = fist held for 250 ms, released, then held 250 ms again within window
const unsigned long STOP_DWELL_MS   = 500;   // hold time to register Stop
const unsigned long BLINK_DWELL_MS  = 250;   // hold time per tap to count toward Blink
const unsigned long BLINK_WINDOW_MS = 1500;  // window to land both taps in

int           blinkTaps      = 0;   // number of valid short holds counted
unsigned long firstTapMs     = 0;   // time of first blink tap

// ---------------- Problem shake tracking ----------------
// Flat hand (stop pose) rocked side to side — gForceAY oscillates +/-
// Swing range: ~-0.8 to ~+0.8
const float   SHAKE_HIGH        =  0.50;  // AY above this = rocked one way
const float   SHAKE_LOW         = -0.50;  // AY below this = rocked other way
const int     SHAKE_ROCKS_NEEDED = 4;     // threshold crossings needed (2 full rocks)
const unsigned long SHAKE_WINDOW_MS = 2000;

int           shakeCount     = 0;   // number of threshold crossings so far
int           shakeDir       = 0;   // last crossing direction: +1 or -1
unsigned long shakeStartMs   = 0;   // time of first crossing

// ---------------- UI/status ----------------
unsigned long lastStatusMs = 0;
const unsigned long STATUS_INTERVAL_MS = 1000;

// ---------------- Serial input buffer ----------------
String inputBuffer = "";

// ============================================================
// Helpers
// ============================================================
bool isAllDigits(String s) {
  if (s.length() == 0) return false;
  for (unsigned int i = 0; i < s.length(); i++) {
    if (!isDigit(s[i])) return false;
  }
  return true;
}

// ============================================================
// Read flex sensors with EMA filtering
// ============================================================
void readFiltered() {
  f0 = (1.0 - ALPHA) * f0 + ALPHA * analogRead(A0);
  f1 = (1.0 - ALPHA) * f1 + ALPHA * analogRead(A1);
  f2 = (1.0 - ALPHA) * f2 + ALPHA * analogRead(A2);
  f3 = (1.0 - ALPHA) * f3 + ALPHA * analogRead(A3);
  f4 = (1.0 - ALPHA) * f4 + ALPHA * analogRead(A4);

  sensor  = (int)f0;
  sensor1 = (int)f1;
  sensor2 = (int)f2;
  sensor3 = (int)f3;
  sensor4 = (int)f4;
}

// ============================================================
// Calibrate relaxed hand (minimum values)   Command: c
// ============================================================
void calibrateRelaxed() {
  Serial.println("# Calibrating relaxed hand - keep hand relaxed and still...");

  float sum[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 200; i++) {
    readFiltered();
    sum[0] += sensor;  sum[1] += sensor1;
    sum[2] += sensor2; sum[3] += sensor3;
    sum[4] += sensor4;
    delay(5);
  }
  for (int i = 0; i < 5; i++) flexMin[i] = sum[i] / 200.0;

  minCalibrated = true;
  Serial.print("# Relaxed calibration complete. Min: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(flexMin[i], 1);
    if (i < 4) Serial.print(", ");
  }
  Serial.println();
}

// ============================================================
// Calibrate fist (maximum values)           Command: x
// ============================================================
void calibrateFist() {
  Serial.println("# Calibrating fist - close fist and hold still...");

  float sum[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < 200; i++) {
    readFiltered();
    sum[0] += sensor;  sum[1] += sensor1;
    sum[2] += sensor2; sum[3] += sensor3;
    sum[4] += sensor4;
    delay(5);
  }
  for (int i = 0; i < 5; i++) flexMax[i] = sum[i] / 200.0;

  maxCalibrated = true;
  Serial.print("# Fist calibration complete. Max: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(flexMax[i], 1);
    if (i < 4) Serial.print(", ");
  }
  Serial.println();
}

// ============================================================
// Normalize flex values to 0..1
//   0.0 = fully relaxed, 1.0 = fully bent (fist)
// ============================================================
void normalizeFlex() {
  int raw[5] = {sensor, sensor1, sensor2, sensor3, sensor4};
  for (int i = 0; i < 5; i++) {
    float range = flexMax[i] - flexMin[i];
    if (range < 1.0) range = 1.0;
    float value = (raw[i] - flexMin[i]) / range;
    if (value < 0.0) value = 0.0;
    if (value > 1.0) value = 1.0;
    flexNorm[i] = value;
  }
}

// ============================================================
// Gesture conditions (normalized 0..1)
//
// flexNorm[0] = thumb  (A0)
// flexNorm[1] = index  (A1)
// flexNorm[2] = middle (A2)
// flexNorm[3] = ring   (A3)
// flexNorm[4] = pinky  (A4)
//
// 0.0 = straight, 1.0 = fully curled
//
// These were translated from the original raw thresholds.
// Re-tune after running a real calibration on your glove.
// ============================================================
bool gestureOK() {
  // index curled, others straight; original: s1>290,s2<175,s3<240,s4<190,s0>300
  return (flexNorm[1] > 0.45 &&
          flexNorm[2] < 0.6 &&
          flexNorm[3] < 0.25 &&
          flexNorm[4] < 0.25 &&
          flexNorm[0] > 0.60);
}

bool gestureGun() {
  // index+thumb extended, others curled; original: s1<220,s2>270,s3>350,s4>310,s0<250
  return (flexNorm[1] < 0.45 &&
          flexNorm[2] > 0.8 &&
          flexNorm[3] > 0.85 &&
          flexNorm[4] > 0.60 &&
          flexNorm[0] < 0.25);
}

bool gestureStop() {
  // fist — all fingers curled
  return (flexNorm[0] > 0.60 &&
          flexNorm[1] > 0.60 &&
          flexNorm[2] > 0.60 &&
          flexNorm[3] > 0.60 &&
          flexNorm[4] > 0.60);
}

bool gestureFlatHand() {
  // all fingers extended — used for Problem shake
  return (flexNorm[0] < 0.50 &&
          flexNorm[1] < 0.50 &&
          flexNorm[2] < 0.50 &&
          flexNorm[3] < 0.50 &&
          flexNorm[4] < 0.50);
}

bool gestureUp() {
  // thumb extended, other four fingers curled, hand tilted up
  return (flexNorm[0] < 0.30 &&   // thumb extended
          flexNorm[1] > 0.60 &&   // index curled
          flexNorm[2] > 0.60 &&   // middle curled
          flexNorm[3] > 0.60 &&   // ring curled
          flexNorm[4] > 0.60 &&   // pinky curled
          gForceAY > 0.85);
}

bool gestureDown() {
  // same hand shape, tilted down
  return (flexNorm[0] < 0.30 &&
          flexNorm[1] > 0.60 &&
          flexNorm[2] > 0.60 &&
          flexNorm[3] > 0.60 &&
          flexNorm[4] > 0.60 &&
          gForceAY < -0.4);
}

bool gestureLook() {
  // Index + middle extended (like a V pointing away), others curled, thumb tucked
  // "Look at me" / peace sign directed outward
  return (flexNorm[0] > 0.55 &&   // thumb curled
          flexNorm[1] < 0.6 &&   // index straight
          flexNorm[2] < 0.8 &&   // middle straight
          flexNorm[3] > 0.75 &&   // ring curled
          flexNorm[4] > 0.7 &&
          gForceAX > 0.60);    // pinky curled
}

bool gesturePhoto() {
  // Thumb and index form a rectangle/frame, others curled
  // Thumb and index extended, middle/ring/pinky curled
  return (flexNorm[0] < 0.30 &&   // thumb extended
          flexNorm[1] < 0.45 &&   // index extended
          flexNorm[2] < 0.55 &&   // middle curled
          flexNorm[3] > 0.8 &&   // ring curled
          flexNorm[4] > 0.6);    // pinky curled
}

bool gestureCool() {
  // Thumb + pinky extended, index/middle/ring curled (shaka / hang loose)
  return (flexNorm[0] < 0.30 &&   // thumb extended
          flexNorm[1] > 0.60 &&   // index curled
          flexNorm[2] > 0.60 &&   // middle curled
          flexNorm[3] > 0.60 &&   // ring curled
          flexNorm[4] < 0.30);    // pinky extended
}

bool gestureCut() {
  // Index + middle extended and together (scissors), thumb/ring/pinky curled
  return (flexNorm[0] > 0.55 &&   // thumb curled
          flexNorm[1] < 0.6 &&   // index straight
          flexNorm[2] < 0.8 &&   // middle straight
          flexNorm[3] > 0.75 &&   // ring curled
          flexNorm[4] > 0.7 &&
          gForceAX > -0.100 && gForceAX < 0.150);
  // Note: Cut and Look use the same finger pose — distinguish them by
  // orienting the hand differently (e.g. palm facing self vs outward)
  // and adding a gForce condition once you know your axis values.
}

bool gestureRock() {
  // Index + pinky extended, middle/ring curled, thumb tucked (devil horns)
  return (flexNorm[0] > 0.55 &&   // thumb curled
          flexNorm[1] < 0.25 &&   // index straight
          flexNorm[2] > 0.60 &&   // middle curled
          flexNorm[3] > 0.60 &&   // ring curled
          flexNorm[4] < 0.25);    // pinky straight
}

bool gesturePoint() {
  // Index extended, all others curled including thumb
  return (flexNorm[0] > 0.55 &&   // thumb curled
          flexNorm[1] < 0.5 &&   // index straight
          flexNorm[2] > 0.60 &&   // middle curled
          flexNorm[3] > 0.60 &&   // ring curled
          flexNorm[4] > 0.60);    // pinky curled
}

// ============================================================
// Print current normalized values (debug)
// ============================================================
void printNorm() {
  Serial.print("# norm: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(flexNorm[i], 3);
    if (i < 4) Serial.print(", ");
  }
  Serial.print("  gX="); Serial.print(gForceAX, 3);
  Serial.print("  gY="); Serial.print(gForceAY, 3);
  Serial.print("  gZ="); Serial.println(gForceAZ, 3);
}

// ============================================================
// Motor pulse helpers
// ============================================================

// Call this to start N vibration bursts (non-blocking)
void motorPulse(int pulses) {
  motorPulsesLeft = pulses;
  motorOn         = true;
  motorOnMs       = millis();
  digitalWrite(motorPin, HIGH);
}

void motorStop() {
  motorPulsesLeft = 0;
  motorOn         = false;
  digitalWrite(motorPin, LOW);
}

// Call every loop() iteration to drive the pulse state machine
void updateMotor() {
  if (motorPulsesLeft <= 0) return;

  unsigned long now = millis();

  if (motorOn) {
    if (now - motorOnMs >= MOTOR_ON_MS) {
      digitalWrite(motorPin, LOW);
      motorOn    = false;
      motorOffMs = now;
      motorPulsesLeft--;
    }
  } else {
    if (motorPulsesLeft > 0 && now - motorOffMs >= MOTOR_OFF_MS) {
      digitalWrite(motorPin, HIGH);
      motorOn   = true;
      motorOnMs = now;
    }
  }
}

// ============================================================
// Problem shake tracker — runs every loop() independently of
// the latch state machine so it can accumulate crossings freely.
// Fires GESTURE: Problem and resets when enough rocks detected.
// ============================================================
void updateShake() {
  // Looser flat-hand check than gestureStop() — fingers just need to be
  // mostly extended, not perfectly, since shaking causes some flex noise
  bool flatHand = gestureFlatHand();

  if (!flatHand) {
    shakeCount   = 0;
    shakeDir     = 0;
    shakeStartMs = 0;
    return;
  }

  // Expire window
  if (shakeCount > 0 && millis() - shakeStartMs > SHAKE_WINDOW_MS) {
    shakeCount   = 0;
    shakeDir     = 0;
    shakeStartMs = 0;
  }

  // Detect a new crossing in the opposite direction
  int newDir = 0;
  if      (gForceAY > SHAKE_HIGH) newDir =  1;
  else if (gForceAY < SHAKE_LOW)  newDir = -1;

  if (newDir != 0 && newDir != shakeDir) {
    if (shakeCount == 0) shakeStartMs = millis();
    shakeCount++;
    shakeDir = newDir;

    if (shakeCount >= SHAKE_ROCKS_NEEDED) {
      Serial.println("GESTURE: Problem");
      printNorm();
      motorPulse(4);
      shakeCount   = 0;
      shakeDir     = 0;
      shakeStartMs = 0;
    }
  }
}

// ============================================================
// Gesture detection state machine
// ============================================================
void handleDetect() {
  if (!latched) {
    // Determine which gesture (if any) is currently held
    GestureID_t candidate = G_NONE;
    if      (gestureOK()    ) candidate = G_OK;
    else if (gestureGun()   ) candidate = G_GUN;
    else if (gestureUp()    ) candidate = G_UP;
    else if (gestureDown()  ) candidate = G_DOWN;
    else if (gestureLook()  ) candidate = G_LOOK;
    else if (gesturePhoto() ) candidate = G_PHOTO;
    else if (gestureCool()  ) candidate = G_COOL;
    else if (gestureCut()   ) candidate = G_CUT;
    else if (gestureRock()  ) candidate = G_ROCK;
    else if (gestureStop()  ) candidate = G_STOP;
    else if (gesturePoint()  ) candidate = G_POINT;

    if (candidate != G_NONE) {
      if (holdStartMs == 0) holdStartMs = millis();
      unsigned long heldMs = millis() - holdStartMs;

      if (candidate == G_STOP) {
        // Expire blink window if too much time has passed since first tap
        if (blinkTaps > 0 && millis() - firstTapMs > BLINK_WINDOW_MS) {
          blinkTaps = 0;
          firstTapMs = 0;
        }

        if (heldMs >= STOP_DWELL_MS && blinkTaps == 0) {
          // Long hold with no prior tap → Stop
          Serial.println("GESTURE: Stop");
          printNorm();
          motorPulse(3);
          latched = true;
          activeGesture = G_STOP;
        }
        // Short holds (< STOP_DWELL_MS) are counted on release — see else branch below

      } else {
        // All other gestures fire after standard dwell
        if (heldMs >= DWELL_MS) {
          switch (candidate) {
            case G_OK:    Serial.println("GESTURE: OK");    break;
            case G_GUN:   Serial.println("GESTURE: Gun");   break;
            case G_UP:    Serial.println("GESTURE: Up");    break;
            case G_DOWN:  Serial.println("GESTURE: Down");  break;
            case G_LOOK:  Serial.println("GESTURE: Look");  break;
            case G_PHOTO: Serial.println("GESTURE: Photo"); break;
            case G_COOL:  Serial.println("GESTURE: Cool");  break;
            case G_CUT:   Serial.println("GESTURE: Cut");   break;
            case G_ROCK:  Serial.println("GESTURE: Rock");  break;
            case G_POINT:  Serial.println("GESTURE: Point");  break;
            default: break;
          }
          printNorm();
          motorPulse(3);
          latched = true;
          activeGesture = candidate;
          blinkTaps  = 0;   // non-Stop gesture resets blink counter
          firstTapMs = 0;
        }
      }

    } else {
      // Pose just dropped — check if a short Stop hold should count as a blink tap
      if (holdStartMs != 0) {
        unsigned long heldMs = millis() - holdStartMs;

        if (heldMs >= BLINK_DWELL_MS && heldMs < STOP_DWELL_MS) {
          // Valid short tap — count it
          if (blinkTaps == 0) firstTapMs = millis();
          blinkTaps++;
          motorPulse(1);  // single buzz: tap acknowledged

          if (blinkTaps >= 2) {
            // Second tap within window → Blink!
            Serial.println("GESTURE: Blink");
            printNorm();
            motorPulse(5);
            blinkTaps  = 0;
            firstTapMs = 0;
          }
        }
      }
      holdStartMs = 0;
    }

  } else {
    // Latched: wait for the pose to be released before rearming
    bool stillHeld = false;
    switch (activeGesture) {
      case G_OK:    stillHeld = gestureOK();    break;
      case G_GUN:   stillHeld = gestureGun();   break;
      case G_STOP:
      case G_BLINK: stillHeld = gestureStop();  break;
      case G_UP:    stillHeld = gestureUp();    break;
      case G_DOWN:  stillHeld = gestureDown();  break;
      case G_LOOK:  stillHeld = gestureLook();  break;
      case G_PHOTO: stillHeld = gesturePhoto(); break;
      case G_COOL:  stillHeld = gestureCool();  break;
      case G_CUT:   stillHeld = gestureCut();   break;
      case G_ROCK:  stillHeld = gestureRock();  break;
      case G_POINT:  stillHeld = gesturePoint();  break;
      default: break;
    }

    if (!stillHeld) {
      if (releaseStartMs == 0) releaseStartMs = millis();
      if (millis() - releaseStartMs >= REARM_MS) {
        Serial.println("# Ready");
        motorStop();
        latched        = false;
        activeGesture  = G_NONE;
        holdStartMs    = 0;
        releaseStartMs = 0;
      }
    } else {
      releaseStartMs = 0;  // pose came back, cancel release timer
    }
  }
}

// ============================================================
// CSV logging helpers
// ============================================================
void setSampleHz(unsigned long hz) {
  if (hz < 1)   hz = 1;
  if (hz > 200) hz = 200;
  sampleHz       = hz;
  samplePeriodUs = 1000000UL / sampleHz;
  Serial.print("# Sample rate set to ");
  Serial.print(sampleHz);
  Serial.println(" Hz");
}

void printCSVHeader() {
  Serial.println("time_ms,raw0,raw1,raw2,raw3,raw4,gyroX_dps,gyroY_dps,gyroZ_dps,norm0,norm1,norm2,norm3,norm4,gestureID,Button");
}

void logCSVRow() {
  unsigned long t = millis() - logStartMs;
  Serial.print(t);           Serial.print(",");
  Serial.print(sensor);      Serial.print(",");
  Serial.print(sensor1);     Serial.print(",");
  Serial.print(sensor2);     Serial.print(",");
  Serial.print(sensor3);     Serial.print(",");
  Serial.print(sensor4);     Serial.print(",");
  Serial.print(dpsGX, 3);   Serial.print(",");
  Serial.print(dpsGY, 3);   Serial.print(",");
  Serial.print(dpsGZ, 3);   Serial.print(",");
  Serial.print(flexNorm[0], 4); Serial.print(",");
  Serial.print(flexNorm[1], 4); Serial.print(",");
  Serial.print(flexNorm[2], 4); Serial.print(",");
  Serial.print(flexNorm[3], 4); Serial.print(",");
  Serial.print(flexNorm[4], 4); Serial.print(",");
  Serial.print(gestureID);   Serial.print(",");
  Serial.println(softButton ? 1 : 0);
}

void startLogging() {
  currentMode  = MODE_LOG;
  logStartMs   = millis();
  lastLogUs    = micros();
  Serial.println("# Switching to LOG mode. Starting CSV output...");
  printCSVHeader();
}

void startDetecting() {
  currentMode    = MODE_DETECT;
  latched        = false;
  activeGesture  = G_NONE;
  holdStartMs    = 0;
  releaseStartMs = 0;
  Serial.println("# Switching to DETECT mode. Gesture recognition active.");
}

// ============================================================
// Serial command processing
//
//  c       = calibrate relaxed hand
//  x       = calibrate fist
//  d       = start gesture detection mode  (default)
//  s       = start CSV logging mode
//  h50     = set log sample rate to 50 Hz
//  12      = set gesture label (LOG mode only)
//  n       = print current normalized values
// ============================================================
void processLineCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // h50 / hz50 — sample rate
  if (cmd[0] == 'h' || cmd[0] == 'H') {
    String numPart = cmd.substring(1);
    if (numPart.length() >= 1 && (numPart[0] == 'z' || numPart[0] == 'Z'))
      numPart = numPart.substring(1);
    numPart.trim();
    if (isAllDigits(numPart)) setSampleHz(numPart.toInt());
    else Serial.println("# Invalid sample rate. Example: h50");
    return;
  }

  // pure number = gesture label (LOG mode)
  if (isAllDigits(cmd)) {
    if (currentMode != MODE_LOG) {
      Serial.println("# Start logging first with s");
      return;
    }
    gestureID = cmd.toInt();
    Serial.print("# Gesture ID set to ");
    Serial.println(gestureID);
    return;
  }

  Serial.println("# Unknown command");
}

void handleSerialInput() {
  while (Serial.available() > 0) {
    char ch = Serial.read();
    if (ch == '\r') continue;

    if (ch == '\n') {
      if (inputBuffer.length() > 0) {
        processLineCommand(inputBuffer);
        inputBuffer = "";
      }
      continue;
    }

    // Immediate single-letter commands when buffer is empty
    if (inputBuffer.length() == 0) {
      if (ch == 'c' || ch == 'C') { calibrateRelaxed(); continue; }
      if (ch == 'x' || ch == 'X') { calibrateFist();    continue; }

      if (ch == 's' || ch == 'S') {
        if (minCalibrated && maxCalibrated) startLogging();
        else Serial.println("# Calibration incomplete. Press c then x first.");
        continue;
      }

      if (ch == 'd' || ch == 'D') {
        if (minCalibrated && maxCalibrated) startDetecting();
        else Serial.println("# Calibration incomplete. Press c then x first.");
        continue;
      }

      if (ch == 'n' || ch == 'N') { printNorm(); continue; }

      if (ch == 'b' || ch == 'B') {
        softButton = !softButton;
        // Serial.print("# Button ");
        // Serial.println(softButton ? "ON (1)" : "OFF (0)");
        continue;
      }
    }

    inputBuffer += ch;
  }
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(button, INPUT_PULLUP);
  pinMode(motorPin, OUTPUT);
  while (!Serial) { ; }
  delay(2000);

  Serial.println("# Underwater glove - combined logger + gesture detector");
  Serial.println("# Commands:");
  Serial.println("#   c       = calibrate relaxed hand");
  Serial.println("#   x       = calibrate fist");
  Serial.println("#   d       = start DETECT mode (gesture recognition)");
  Serial.println("#   s       = start LOG mode (CSV output)");
  Serial.println("#   h50     = set log sample rate to 50 Hz");
  Serial.println("#   12      = set gesture label (LOG mode only)");
  Serial.println("#   n       = print normalized values now");
  Serial.println("#   b       = toggle button ON/OFF in CSV log");
  Serial.println("# Workflow: c -> x -> d (detect) or s (log)");
}

// ============================================================
// Main loop
// ============================================================
void loop() {
  handleSerialInput();
  wakeSensor(MPU_ADDRESS);

  readFiltered();
  readGyroData(MPU_ADDRESS, rawGX, rawGY, rawGZ);
  rawGyroToDPS(rawGX, rawGY, rawGZ, dpsGX, dpsGY, dpsGZ);
  readAccelData(MPU_ADDRESS, rawAX, rawAY, rawAZ);
  rawAccelToGForce(rawAX, rawAY, rawAZ, gForceAX, gForceAY, gForceAZ);
  normalizeFlex();

  // Status before calibration is done
  if (!minCalibrated || !maxCalibrated) {
    unsigned long nowMs = millis();
    if (nowMs - lastStatusMs >= STATUS_INTERVAL_MS) {
      lastStatusMs = nowMs;
      Serial.print("# Waiting for calibration: minCal=");
      Serial.print(minCalibrated);
      Serial.print(" maxCal=");
      Serial.print(maxCalibrated);
      Serial.println("  -> press c then x");
    }
    return;  // don't run detection/logging until calibrated
  }

  if (currentMode == MODE_DETECT) {
    updateMotor();
    updateShake();
    handleDetect();
    delay(10);

  } else if (currentMode == MODE_LOG) {
    unsigned long nowUs = micros();
    if (nowUs - lastLogUs >= samplePeriodUs) {
      lastLogUs += samplePeriodUs;
      logCSVRow();
    }
  }
}
