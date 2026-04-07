// Stepper speed / move control via Serial for Seeed XIAO RP2040 + ISDT17-20
// Motor: 17HS19-1684S-PG5.18 (5.18:1 gearbox, 1.8 deg motor step)
//
// Commands:
//   <rpm>     = continuous output shaft speed
//   r<rot>    = do N output shaft rotations
//   s         = stop with acceleration ramp
//   m<rpm>    = set cruise/max RPM cap
//   a<rate>   = set acceleration in output RPM/s


#define PIN_STP  D10
#define PIN_DIR  D9
#define PIN_EN   D8

#define MICROSTEPS        4
#define MOTOR_STEPS_REV   200.0f
#define GEAR_RATIO        5.18f
#define MOTOR_HARD_MAX_RPM 1000.0f

// Output shaft microsteps per revolution:
#define STEPS_PER_REV_F   (MOTOR_STEPS_REV * MICROSTEPS * GEAR_RATIO)
#define STEPS_PER_REV     ((long)(STEPS_PER_REV_F + 0.5f))

#define MAX_PULSE_FREQ    20000.0f   // max step pulse rate in steps/sec
#define MIN_PULSE_WIDTH_US 2

// Output shaft max RPM from max pulse frequency:
#define HW_MAX_RPM        ((MAX_PULSE_FREQ * 60.0f) / STEPS_PER_REV_F)
#define SAFE_MAX_RPM      ((MOTOR_HARD_MAX_RPM) / GEAR_RATIO)
#define EFFECTIVE_MAX_RPM ((HW_MAX_RPM < SAFE_MAX_RPM) ? HW_MAX_RPM : SAFE_MAX_RPM)

// User-facing settings (output shaft units)
float maxRPM = EFFECTIVE_MAX_RPM;
float accelRPMps = 150.0f;   // output shaft RPM/s

// Internal motion state (steps/sec units, signed)
float targetStepRate = 0.0f;
float currentStepRate = 0.0f;
float maxStepRate = 0.0f;
float accelStepRate = 0.0f;

// Step timing
unsigned long stepIntervalUs = 0;
unsigned long lastStepTimeUs = 0;
unsigned long lastRateUpdateUs = 0;

// Move state
bool moveActive = false;
long moveStepsRemaining = 0;
float moveCruiseStepRate = 0.0f;   // signed

// -------------------- conversions --------------------

float rpmToStepsPerSec(float rpm) {
  return rpm * STEPS_PER_REV_F / 60.0f;
}

float stepsPerSecToRPM(float sps) {
  return sps * 60.0f / STEPS_PER_REV_F;
}

void refreshDerivedLimits() {
  maxStepRate = rpmToStepsPerSec(maxRPM);
  accelStepRate = rpmToStepsPerSec(accelRPMps);
  if (accelStepRate < 1.0f) accelStepRate = 1.0f;
}

// -------------------- step timing --------------------

void recalcStepInterval() {
  unsigned long previousIntervalUs = stepIntervalUs;
  float absRate = fabs(currentStepRate);

  if (absRate < 0.5f) {
    stepIntervalUs = 0;
    return;
  }

  if (absRate > MAX_PULSE_FREQ) absRate = MAX_PULSE_FREQ;

  stepIntervalUs = (unsigned long)(1000000.0f / absRate);
  if (stepIntervalUs < 1) stepIntervalUs = 1;

  if (previousIntervalUs == 0) {
    lastStepTimeUs = micros();
  }
}

void emitStepPulse() {
  digitalWrite(PIN_STP, HIGH);
  delayMicroseconds(MIN_PULSE_WIDTH_US);
  digitalWrite(PIN_STP, LOW);
}

// -------------------- motion planning --------------------

long decelStepsFromRate(float stepRate) {
  float v = fabs(stepRate);
  float d = (v * v) / (2.0f * accelStepRate);
  return (long)(d + 1.0f);
}

void finishMove() {
  moveActive = false;
  currentStepRate = 0.0f;
  targetStepRate = 0.0f;
  recalcStepInterval();
}

void updateMovePlanner() {
  if (!moveActive) return;

  if (moveStepsRemaining <= 0) {
    targetStepRate = 0.0f;
    return;
  }

  long stopDist = decelStepsFromRate(currentStepRate);

  if (moveStepsRemaining <= stopDist && fabs(currentStepRate) >= 0.5f) {
    targetStepRate = 0.0f;
  } else {
    targetStepRate = moveCruiseStepRate;
  }
}

void updateStepRate() {
  unsigned long nowUs = micros();
  float dt = (nowUs - lastRateUpdateUs) * 1e-6f;
  lastRateUpdateUs = nowUs;

  if (dt > 0.05f) dt = 0.05f;
  if (dt <= 0.0f) return;

  float deltaMax = accelStepRate * dt;
  float error = targetStepRate - currentStepRate;

  if (error > deltaMax) {
    currentStepRate += deltaMax;
  } else if (error < -deltaMax) {
    currentStepRate -= deltaMax;
  } else {
    currentStepRate = targetStepRate;
  }

  // Clamp to hardware limit just in case
  if (currentStepRate > MAX_PULSE_FREQ) currentStepRate = MAX_PULSE_FREQ;
  if (currentStepRate < -MAX_PULSE_FREQ) currentStepRate = -MAX_PULSE_FREQ;

  digitalWrite(PIN_DIR, currentStepRate >= 0.0f ? HIGH : LOW);
  recalcStepInterval();

  // Complete move only when both:
  // 1) no more commanded steps remain
  // 2) speed has ramped to zero
  if (moveActive && moveStepsRemaining <= 0 &&
      fabs(currentStepRate) < 0.5f &&
      fabs(targetStepRate) < 0.5f) {
    finishMove();
  }
}

// -------------------- serial --------------------

void printStatus() {
  Serial.print("Max RPM: ");
  Serial.print(maxRPM, 3);
  Serial.print(" | Accel RPM/s: ");
  Serial.print(accelRPMps, 3);
  Serial.print(" | Steps/rev: ");
  Serial.print(STEPS_PER_REV);
  Serial.print(" | Safe max RPM: ");
  Serial.print(EFFECTIVE_MAX_RPM, 3);
  Serial.print(" | HW max RPM: ");
  Serial.print(HW_MAX_RPM, 3);
  Serial.print(" | Motor hard max RPM: ");
  Serial.println(MOTOR_HARD_MAX_RPM, 3);
}

void startRotationMove(float rotations) {
  if (rotations == 0.0f) return;

  moveActive = true;
  moveStepsRemaining = (long)(fabs(rotations) * STEPS_PER_REV_F + 0.5f);
  moveCruiseStepRate = (rotations > 0.0f) ? maxStepRate : -maxStepRate;

  targetStepRate = moveCruiseStepRate;

  Serial.print("Move: ");
  Serial.print(rotations, 5);
  Serial.print(" rev (");
  Serial.print(moveStepsRemaining);
  Serial.print(" steps) at ");
  Serial.print(maxRPM, 3);
  Serial.println(" RPM");
}

void setContinuousRPM(float rpm) {
  // cancel move mode
  moveActive = false;
  moveStepsRemaining = 0;

  if (rpm > maxRPM) rpm = maxRPM;
  if (rpm < -maxRPM) rpm = -maxRPM;

  targetStepRate = rpmToStepsPerSec(rpm);

  Serial.print("Target RPM: ");
  Serial.println(rpm, 3);
}

void stopMotion() {
  moveActive = false;
  moveStepsRemaining = 0;
  targetStepRate = 0.0f;
  Serial.println("Stopping.");
}

void processSerial() {
  static char buf[24];
  static uint8_t idx = 0;

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (idx == 0) continue;

      buf[idx] = '\0';
      idx = 0;

      if (buf[0] == 'r' || buf[0] == 'R') {
        float rotations = atof(buf + 1);
        startRotationMove(rotations);

      } else if ((buf[0] == 's' || buf[0] == 'S') && buf[1] == '\0') {
        stopMotion();

      } else if (buf[0] == 'm' || buf[0] == 'M') {
        float val = atof(buf + 1);
        if (val > 0.0f) {
          if (val > EFFECTIVE_MAX_RPM) val = EFFECTIVE_MAX_RPM;
          maxRPM = val;
          refreshDerivedLimits();

          // keep active setpoints within updated max
          if (targetStepRate > maxStepRate) targetStepRate = maxStepRate;
          if (targetStepRate < -maxStepRate) targetStepRate = -maxStepRate;
          if (moveActive) {
            moveCruiseStepRate = (moveCruiseStepRate >= 0.0f) ? maxStepRate : -maxStepRate;
          }

          Serial.print("Max RPM set to ");
          Serial.println(maxRPM, 3);
        }

      } else if (buf[0] == 'a' || buf[0] == 'A') {
        float val = atof(buf + 1);
        if (val > 0.0f) {
          accelRPMps = val;
          refreshDerivedLimits();

          Serial.print("Accel set to ");
          Serial.print(accelRPMps, 3);
          Serial.println(" RPM/s");
        }

      } else {
        float val = atof(buf);
        setContinuousRPM(val);
      }

    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }
}

// -------------------- setup / loop --------------------

void setup() {
  pinMode(PIN_STP, OUTPUT);
  pinMode(PIN_DIR, OUTPUT);
  pinMode(PIN_EN, OUTPUT);

  digitalWrite(PIN_EN, LOW);
  digitalWrite(PIN_DIR, HIGH);
  digitalWrite(PIN_STP, LOW);

  Serial.begin(115200);
  while (!Serial) delay(10);

  refreshDerivedLimits();

  Serial.println("Stepper ready.");
  Serial.print("Gear ratio: ");
  Serial.println(GEAR_RATIO, 3);
  Serial.print("Steps/output rev: ");
  Serial.println(STEPS_PER_REV);
  Serial.print("Motor hard max RPM: ");
  Serial.println(MOTOR_HARD_MAX_RPM, 3);
  Serial.print("Safe output max RPM: ");
  Serial.println(EFFECTIVE_MAX_RPM, 3);
  Serial.println("Commands:");
  Serial.println("  <rpm>   = continuous speed");
  Serial.println("  r<rot>  = relative move");
  Serial.println("  s       = stop with ramp");
  Serial.println("  m<rpm>  = max RPM cap");
  Serial.println("  a<rate> = accel in RPM/s");

  printStatus();

  unsigned long now = micros();
  lastRateUpdateUs = now;
  lastStepTimeUs = now;
}

void loop() {
  processSerial();
  updateMovePlanner();
  updateStepRate();

  if (stepIntervalUs > 0) {
    unsigned long nowUs = micros();

    while ((unsigned long)(nowUs - lastStepTimeUs) >= stepIntervalUs) {
      lastStepTimeUs += stepIntervalUs;

      emitStepPulse();

      if (moveActive && moveStepsRemaining > 0) {
        moveStepsRemaining--;

        if (moveStepsRemaining <= 0) {
          moveStepsRemaining = 0;
          targetStepRate = 0.0f;
          if (fabs(currentStepRate) < 0.5f) {
            finishMove();
            break;
          }
        }
      }

      nowUs = micros();
    }
  }
}
