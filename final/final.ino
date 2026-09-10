#include "mpu6050.h"

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

// "Button" logged in CSV — toggled only by the 'b' key in Serial Monitor,
// no physical button wired in.
bool button = false;

// EMA filtered values
float f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
const float ALPHA = 0.2;

// ---------------- Calibration + normalization ----------------
float flexMin[5]  = {0, 0, 0, 0, 0};
float flexMax[5]  = {1, 1, 1, 1, 1};
float flexNorm[5] = {0, 0, 0, 0, 0};

bool minCalibrated = false;
bool maxCalibrated = false;

// ---------------- Orientation reference ----------------
// Unit gravity vector captured during 'c' with the hand held open and flat,
// BACK OF THE HAND FACING UP (palm facing down). That pose is defined as "up";
// everything downstream reports orientation relative to it.
float upRefX = 0, upRefY = 1, upRefZ = 0;
bool  orientCalibrated = false;

// Which way the hand is currently facing, relative to the calibrated pose.
// Refreshed every loop() by updateFacing(); read freely by any function.
enum Facing_t { FACE_UNKNOWN, FACE_UP, FACE_DOWN, FACE_LEFT, FACE_RIGHT,
                FACE_FORWARD, FACE_BACK };
Facing_t handFacing      = FACE_UNKNOWN;
float    handFacingAngle = 0;   // degrees away from the calibrated "up" pose

// ---------------- Operating mode ----------------
// DETECT = gesture recognition; LOG = CSV output
enum Mode { MODE_DETECT, MODE_LOG };
Mode currentMode = MODE_DETECT;

// ---------------- Data logging (LOG mode) ----------------
int gestureID = 0;
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

// Static vs dynamic classification.
//   static  = recognized from finger pose + orientation (handleDetect)
//   dynamic = recognized from hand motion              (handleDynamic)
bool isDynamicGesture(GestureID_t g) {
  return (g == G_PROBLEM);
}
// true when the most recently recognized gesture was a dynamic one
bool gestureDynamic = false;

// "No gesture recognized" state.
// Stays true from startup / from the moment the hand returns to neutral,
// until any gesture is recognized. "GESTURE: None" is printed once on the
// transition back to this state.
bool noGesture = true;

// ---------------- Stop / Blink thresholds ----------------
// Stop  = fist held for 500 ms (single hold)
// Blink = fist held for 250 ms, released, then held 250 ms again within window
const unsigned long STOP_DWELL_MS   = 500;   // hold time to register Stop
const unsigned long BLINK_DWELL_MS  = 250;   // hold time per tap to count toward Blink
const unsigned long BLINK_WINDOW_MS = 1500;  // window to land both taps in

int           blinkTaps      = 0;   // number of valid short holds counted
unsigned long firstTapMs     = 0;   // time of first blink tap

// ============================================================
// Motion analysis  (feeds DYNAMIC gesture recognition)
//
// Static and dynamic detection run every loop() in parallel:
//   - handleDetect()  : finger pose + orientation, needs a still hand
//   - handleDynamic() : gestures made by MOVING the hand
//
// A "dynamic window" is open (dynamicWindow == true) only when ALL of:
//   * the IMU shows significant rotation            (gyroMag > MOTION_GYRO_ON_DPS)
//   * that rotation reverses direction repeatedly   (back-and-forth wave)
//   * the finger pose stays roughly constant        (poseSteady)
// While the hand is moving, static dwell is frozen so a wave is not
// misread as a static pose.
// ============================================================
const float         MOTION_GYRO_ON_DPS    = 90.0;   // |gyro| to enter "moving"
const float         MOTION_GYRO_OFF_DPS   = 45.0;   // |gyro| to leave "moving" (hysteresis)
const float         REVERSAL_HYST_DPS     = 35.0;   // dead-band for counting direction flips
const int           REVERSALS_FOR_DYNAMIC = 3;      // flips inside the window => "repeated"
const unsigned long MOTION_WINDOW_MS      = 1500;   // span the flips must fit within
const float         POSE_DRIFT_TOL        = 0.20;   // max flexNorm wobble to stay "same pose"
const unsigned long POSE_STEADY_MIN_MS    = 200;    // pose must be steady at least this long

// live motion state, refreshed by updateMotion()
float gyroMag       = 0;       // |gyro| magnitude, dps
int   motionAxis    = 0;       // dominant rotation axis: 0=X 1=Y 2=Z
bool  motionActive  = false;   // significant sustained rotation right now
bool  poseSteady    = false;   // fingers ~constant recently
bool  dynamicWindow = false;   // motionActive && poseSteady && repeated pattern
bool  dynAnnounced  = false;   // last-printed search mode (false = static)

// internals
float         flexBaseline[5]    = {0, 0, 0, 0, 0};   // slow EMA of the pose
unsigned long poseSteadySinceMs  = 0;
const int     REV_RING           = 6;
unsigned long revTimes[REV_RING] = {0};
int           revHead            = 0;
int           revFill            = 0;
int           revDir             = 0;

// ---------------- Problem (dynamic gesture) ----------------
// Flat / open hand waved back and forth.
const int     PROBLEM_ROCKS_NEEDED = 4;      // gForceAY sign flips needed
const float   PROBLEM_AY_HIGH      =  0.40;
const float   PROBLEM_AY_LOW       = -0.40;
const unsigned long PROBLEM_WINDOW_MS = 2000;

int           problemRocks   = 0;
int           problemDir     = 0;
unsigned long problemStartMs = 0;

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
  Serial.println("# Calibrating relaxed hand - hold hand OPEN and FLAT, back of hand facing up (palm down), keep still...");

  wakeSensor(MPU_ADDRESS);

  float sum[5]  = {0, 0, 0, 0, 0};
  float gSum[3] = {0, 0, 0};
  for (int i = 0; i < 200; i++) {
    readFiltered();
    sum[0] += sensor;  sum[1] += sensor1;
    sum[2] += sensor2; sum[3] += sensor3;
    sum[4] += sensor4;

    // Sample gravity for the orientation reference
    readAccelData(MPU_ADDRESS, rawAX, rawAY, rawAZ);
    rawAccelToGForce(rawAX, rawAY, rawAZ, gForceAX, gForceAY, gForceAZ);
    gSum[0] += gForceAX; gSum[1] += gForceAY; gSum[2] += gForceAZ;

    delay(5);
  }
  for (int i = 0; i < 5; i++) flexMin[i] = sum[i] / 200.0;

  // Store the mean gravity vector as a unit vector = "hand facing up"
  float gx = gSum[0] / 200.0, gy = gSum[1] / 200.0, gz = gSum[2] / 200.0;
  float mag = sqrt(gx * gx + gy * gy + gz * gz);
  if (mag < 0.1) mag = 1.0;
  upRefX = gx / mag; upRefY = gy / mag; upRefZ = gz / mag;
  orientCalibrated = true;

  minCalibrated = true;
  Serial.print("# Relaxed calibration complete. Min: ");
  for (int i = 0; i < 5; i++) {
    Serial.print(flexMin[i], 1);
    if (i < 4) Serial.print(", ");
  }
  Serial.println();
  Serial.print("# Up reference (g): ");
  Serial.print(upRefX, 3); Serial.print(", ");
  Serial.print(upRefY, 3); Serial.print(", ");
  Serial.println(upRefZ, 3);
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

// Thumb extended, other four fingers curled (thumb-up hand shape).
// Shared by Up and Down; hand orientation tells them apart:
//   Up   = hand facing right
//   Down = hand facing left
bool thumbOnlyPose() {
  return (flexNorm[0] < 0.30 &&   // thumb extended
          flexNorm[1] > 0.60 &&   // index curled
          flexNorm[2] > 0.60 &&   // middle curled
          flexNorm[3] > 0.60 &&   // ring curled
          flexNorm[4] > 0.60);    // pinky curled
}

bool gestureUp() {
  return (thumbOnlyPose() && handFacing == FACE_RIGHT);
}

bool gestureDown() {
  return (thumbOnlyPose() && handFacing == FACE_LEFT);
}

// Index + middle extended, ring + pinky curled, thumb tucked — the same
// finger pose is shared by Look and Cut; hand orientation tells them apart:
//   Look = hand NOT facing left or right
//   Cut  = hand facing left or right
bool twoFingerPose() {
  return (flexNorm[0] > 0.55 &&   // thumb curled
          flexNorm[1] < 0.6 &&    // index straight
          flexNorm[2] < 0.8 &&    // middle straight
          flexNorm[3] > 0.75 &&   // ring curled
          flexNorm[4] > 0.7);     // pinky curled
}

bool gestureLook() {
  return (twoFingerPose() &&
          handFacing != FACE_LEFT &&
          handFacing != FACE_RIGHT);
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
  // Same finger pose as Look (scissors); recognized when the hand is
  // turned to face left or right.
  return (twoFingerPose() &&
          (handFacing == FACE_LEFT || handFacing == FACE_RIGHT));
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

  Serial.print("# motion: looking=");
  Serial.print(dynamicWindow ? "dynamic" : "static");
  Serial.print("  gyroMag="); Serial.print(gyroMag, 0);
  Serial.print("  moving=");   Serial.print(motionActive);
  Serial.print("  poseSteady="); Serial.print(poseSteady);
  Serial.print("  reversals="); Serial.println(revFill);
}

// ============================================================
// Track which way the hand is facing, relative to the pose held during 'c'
// calibration — hand open and flat with the back of the hand facing up
// (palm down). That pose is defined as "up".
//
// Builds an orthonormal frame from the calibrated up-vector, projects the
// current gravity direction onto it, and picks the dominant axis. Called
// every loop() so handFacing / handFacingAngle are always current.
// ============================================================
void updateFacing() {
  if (!orientCalibrated) {
    handFacing      = FACE_UNKNOWN;
    handFacingAngle = 0;
    return;
  }

  // Current gravity as a unit vector
  float m = sqrt(gForceAX * gForceAX + gForceAY * gForceAY + gForceAZ * gForceAZ);
  if (m < 0.1) m = 1.0;
  float gx = gForceAX / m, gy = gForceAY / m, gz = gForceAZ / m;

  // up = calibrated reference axis
  float ux = upRefX, uy = upRefY, uz = upRefZ;

  // Helper axis least parallel to up, then Gram-Schmidt for right / forward
  float hx = 0, hy = 0, hz = 1;
  if (fabs(uz) > 0.9) { hx = 1; hy = 0; hz = 0; }

  // right = up x helper  (normalized)
  float rx = uy * hz - uz * hy;
  float ry = uz * hx - ux * hz;
  float rz = ux * hy - uy * hx;
  float rm = sqrt(rx * rx + ry * ry + rz * rz);
  if (rm < 1e-3) rm = 1.0;
  rx /= rm; ry /= rm; rz /= rm;

  // forward = right x up
  float fx = ry * uz - rz * uy;
  float fy = rz * ux - rx * uz;
  float fz = rx * uy - ry * ux;

  // Components of current gravity in the calibrated frame
  float cu = gx * ux + gy * uy + gz * uz;   // +up  / -down
  float cr = gx * rx + gy * ry + gz * rz;   // +right / -left
  float cf = gx * fx + gy * fy + gz * fz;   // +forward / -back

  float au = fabs(cu), ar = fabs(cr), af = fabs(cf);

  if      (au >= ar && au >= af) handFacing = (cu >= 0) ? FACE_UP      : FACE_DOWN;
  else if (ar >= af)             handFacing = (cr >= 0) ? FACE_RIGHT   : FACE_LEFT;
  else                           handFacing = (cf >= 0) ? FACE_FORWARD : FACE_BACK;

  handFacingAngle = degrees(acos(constrain(cu, -1.0, 1.0)));
}

const char *facingLabel() {
  switch (handFacing) {
    case FACE_UP:      return "up";
    case FACE_DOWN:    return "down";
    case FACE_LEFT:    return "left";
    case FACE_RIGHT:   return "right";
    case FACE_FORWARD: return "forward";
    case FACE_BACK:    return "back";
    default:           return "unknown";
  }
}

void printFacing() {
  if (handFacing == FACE_UNKNOWN) {
    Serial.println("FACING: unknown (run c to calibrate orientation)");
    return;
  }
  Serial.print("FACING: ");
  Serial.print(facingLabel());
  Serial.print("  (");
  Serial.print(handFacingAngle, 0);
  Serial.println(" deg from calibrated up)");
}

const char *gestureName(GestureID_t g) {
  switch (g) {
    case G_OK:      return "OK";
    case G_GUN:     return "Gun";
    case G_STOP:    return "Stop";
    case G_UP:      return "Up";
    case G_DOWN:    return "Down";
    case G_BLINK:   return "Blink";
    case G_LOOK:    return "Look";
    case G_PHOTO:   return "Photo";
    case G_COOL:    return "Cool";
    case G_CUT:     return "Cut";
    case G_ROCK:    return "Rock";
    case G_PROBLEM: return "Problem";
    case G_POINT:   return "Point";
    default:        return "None";
  }
}

// Single place every recognized gesture is reported.
// Sets the static/dynamic flag, clears the idle state, prints the gesture
// line (tagged static/dynamic), the normalized values, and the facing.
void announceGesture(GestureID_t id) {
  gestureDynamic = isDynamicGesture(id);
  noGesture      = false;
  Serial.print("GESTURE: ");
  Serial.print(gestureName(id));
  Serial.println(gestureDynamic ? "  (dynamic)" : "  (static)");
  printNorm();
  printFacing();
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
// Motion analysis — runs every loop(). Decides whether the hand is
// being moved in a repeated back-and-forth pattern with a steady pose,
// i.e. whether a DYNAMIC gesture should be attempted.
// ============================================================
void updateMotion() {
  unsigned long now = millis();

  // --- significant rotation, with hysteresis ---
  gyroMag = sqrt(dpsGX * dpsGX + dpsGY * dpsGY + dpsGZ * dpsGZ);
  if (!motionActive && gyroMag > MOTION_GYRO_ON_DPS)  motionActive = true;
  if ( motionActive && gyroMag < MOTION_GYRO_OFF_DPS) motionActive = false;

  // --- dominant rotation axis (largest instantaneous rate) ---
  float mx = fabs(dpsGX), my = fabs(dpsGY), mz = fabs(dpsGZ);
  if      (mx >= my && mx >= mz) motionAxis = 0;
  else if (my >= mz)             motionAxis = 1;
  else                           motionAxis = 2;
  float axisRate = (motionAxis == 0) ? dpsGX : (motionAxis == 1) ? dpsGY : dpsGZ;

  // --- count back-and-forth reversals on that axis (ring of timestamps) ---
  int prevIdx = (revHead - 1 + REV_RING) % REV_RING;
  if (revFill > 0 && now - revTimes[prevIdx] > MOTION_WINDOW_MS) {
    revFill = 0;
    revDir  = 0;
  }
  int dir = 0;
  if      (axisRate >  REVERSAL_HYST_DPS) dir =  1;
  else if (axisRate < -REVERSAL_HYST_DPS) dir = -1;
  if (dir != 0 && dir != revDir) {
    revDir = dir;
    revTimes[revHead] = now;
    revHead = (revHead + 1) % REV_RING;
    if (revFill < REV_RING) revFill++;
  }
  bool repeated = false;
  if (revFill >= REVERSALS_FOR_DYNAMIC) {
    int oldIdx = (revHead - REVERSALS_FOR_DYNAMIC + REV_RING) % REV_RING;
    if (now - revTimes[oldIdx] <= MOTION_WINDOW_MS) repeated = true;
  }

  // --- finger pose stability: instantaneous vs a slow EMA baseline ---
  float drift = 0;
  for (int i = 0; i < 5; i++) {
    flexBaseline[i] += 0.02f * (flexNorm[i] - flexBaseline[i]);
    float d = fabs(flexNorm[i] - flexBaseline[i]);
    if (d > drift) drift = d;
  }
  if (drift < POSE_DRIFT_TOL) {
    if (poseSteadySinceMs == 0) poseSteadySinceMs = now;
  } else {
    poseSteadySinceMs = 0;
  }
  poseSteady = (poseSteadySinceMs != 0 && now - poseSteadySinceMs >= POSE_STEADY_MIN_MS);

  dynamicWindow = motionActive && poseSteady && repeated;

  // Announce which kind of gesture the system is currently looking for
  if (dynamicWindow != dynAnnounced) {
    dynAnnounced = dynamicWindow;
    Serial.println(dynamicWindow ? "LOOKING: dynamic gesture"
                                 : "LOOKING: static gesture");
  }
}

// ============================================================
// Dynamic gesture recognition — runs every loop() in parallel with
// handleDetect(). Each dynamic gesture is only attempted while a
// dynamic window is open (see updateMotion). Add new dynamic gestures
// as extra blocks below.
// ============================================================
void handleDynamic() {
  if (!dynamicWindow) {
    problemRocks   = 0;
    problemDir     = 0;
    problemStartMs = 0;
    return;
  }

  unsigned long now = millis();

  // ---- Problem: flat / open hand waved back and forth ----
  if (gestureFlatHand()) {
    if (problemStartMs == 0) problemStartMs = now;

    int dir = 0;
    if      (gForceAY > PROBLEM_AY_HIGH) dir =  1;
    else if (gForceAY < PROBLEM_AY_LOW)  dir = -1;

    if (dir != 0 && dir != problemDir) {
      problemDir = dir;
      problemRocks++;

      if (problemRocks >= PROBLEM_ROCKS_NEEDED) {
        announceGesture(G_PROBLEM);
        motorPulse(4);
        problemRocks   = 0;
        problemDir     = 0;
        problemStartMs = 0;
      }
    }

    if (problemStartMs != 0 && now - problemStartMs > PROBLEM_WINDOW_MS) {
      problemRocks   = 0;
      problemDir     = 0;
      problemStartMs = 0;
    }
  } else {
    problemRocks   = 0;
    problemDir     = 0;
    problemStartMs = 0;
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

    // Static gestures need a still hand: while the hand is moving, freeze
    // static dwell and let handleDynamic() run instead.
    if (motionActive) {
      candidate   = G_NONE;
      holdStartMs  = 0;
    }

    // Hand is back to neutral (no pose, no hold underway) — announce once
    // and hold the "no gesture recognized" state until something fires.
    if (candidate == G_NONE && holdStartMs == 0 && !noGesture && !motionActive) {
      noGesture = true;
      Serial.println("GESTURE: None");
    }

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
          announceGesture(G_STOP);
          motorPulse(3);
          latched = true;
          activeGesture = G_STOP;
        }
        // Short holds (< STOP_DWELL_MS) are counted on release — see else branch below

      } else {
        // All other gestures fire after standard dwell
        if (heldMs >= DWELL_MS) {
          announceGesture(candidate);
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
            announceGesture(G_BLINK);
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
  Serial.println(button ? 1 : 0);
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
  noGesture      = true;
  motionActive   = false;
  dynamicWindow  = false;
  dynAnnounced   = false;
  revFill        = 0;
  revDir         = 0;
  Serial.println("# Switching to DETECT mode. Gesture recognition active.");
  Serial.println("GESTURE: None");
  Serial.println("LOOKING: static gesture");
}

// ============================================================
// Serial command processing
//
//  c       = calibrate relaxed hand (open, flat, back of hand up / palm down)
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
        button = !button;
        // Serial.print("# Button ");
        // Serial.println(button ? "ON (1)" : "OFF (0)");
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
  pinMode(motorPin, OUTPUT);
  while (!Serial) { ; }
  delay(2000);

  Serial.println("# Underwater glove - combined logger + gesture detector");
  Serial.println("# Commands:");
  Serial.println("#   c       = calibrate relaxed hand (open, flat, back of hand up / palm down)");
  Serial.println("#   x       = calibrate fist");
  Serial.println("#   d       = start DETECT mode (gesture recognition)");
  Serial.println("#   s       = start LOG mode (CSV output)");
  Serial.println("#   h50     = set log sample rate to 50 Hz");
  Serial.println("#   12      = set gesture label (LOG mode only)");
  Serial.println("#   n       = print normalized values + motion state now");
  Serial.println("#   b       = toggle button ON/OFF in CSV log");
  Serial.println("# Workflow: c -> x -> d (detect) or s (log)");
  Serial.println("# DETECT output: GESTURE: <name> (<static|dynamic>) | None, FACING: <dir>, LOOKING: <static|dynamic>");
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
  updateFacing();

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
    updateMotion();      // refresh motion / dynamic-window state
    handleDynamic();     // dynamic gestures  (runs in parallel)
    handleDetect();      // static gestures
    delay(10);

  } else if (currentMode == MODE_LOG) {
    unsigned long nowUs = micros();
    if (nowUs - lastLogUs >= samplePeriodUs) {
      lastLogUs += samplePeriodUs;
      logCSVRow();
    }
  }
}
