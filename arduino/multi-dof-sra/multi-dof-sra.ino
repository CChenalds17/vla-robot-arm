#include <Servo.h>

const uint8_t CMD_HANDSHAKE = 0x00;
const uint8_t RSP_HANDSHAKE = 0xFF;
const uint8_t CMD_MOVE      = 0x01;
const uint8_t CMD_STATUS    = 0x02;
const uint8_t RSP_MOVED     = 0x81;
const uint8_t RSP_STATUS    = 0x82;

const int NUM_DOFS = 1;
int SERVO_PINS[NUM_DOFS] = {3};
Servo armServos[NUM_DOFS];  // Create servo object to control a servo

void setup() {
  for (int i = 0; i < NUM_DOFS; i++) {
    armServos[i].attach(SERVO_PINS[i]);
  }
  Serial.begin(115200);
  // Wait for handshake command
  while (true) {
    if (Serial.available() >= NUM_DOFS + 1) {
      uint8_t cmd = Serial.read();
      uint8_t args[NUM_DOFS];
      for (int i = 0; i < NUM_DOFS; i++) {
        args[i] = Serial.read(); // Ignored
      }
      if (cmd == CMD_HANDSHAKE) {
        // Reply and break out to normal operation
        Serial.write(RSP_HANDSHAKE);
        for (int i = 0; i < NUM_DOFS; i++) {
          Serial.write((uint8_t)0x00);
        }
        break;
      }
    }
  }
  while (Serial.available()) Serial.read(); // Flush leftover bytes after handshake
}

void loop() {
  if (Serial.available() >= NUM_DOFS + 1) {
    uint8_t cmd = Serial.read();
    uint8_t args[NUM_DOFS];
    for (int i = 0; i < NUM_DOFS; i++) {
      args[i] = Serial.read();
    }
    uint8_t angles[NUM_DOFS];

    if (cmd == CMD_MOVE) {
      Serial.write(RSP_MOVED);
      for (int i = 0; i < NUM_DOFS; i++) {
        args[i] = constrain(args[i], 0, 180); // Constrain to correct bounds
        armServos[i].write(args[i]); // Move servo to input angle
        // Send response
        angles[i] = armServos[i].read();
        Serial.write(angles[i]);
      }
    } else if (cmd == CMD_STATUS) {
      Serial.write(RSP_STATUS);
      for (int i = 0; i < NUM_DOFS; i++) {
        angles[i] = armServos[i].read();
        Serial.write(angles[i]);
      }
    }
  }

}