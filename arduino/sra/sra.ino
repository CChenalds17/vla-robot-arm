#include <Servo.h>
int SERVO_PIN = 3;

Servo myServo;  // Create servo object to control a servo
const uint8_t CMD_HANDSHAKE = 0x00;
const uint8_t RSP_HANDSHAKE = 0xFF;
const uint8_t CMD_MOVE      = 0x01;
const uint8_t CMD_STATUS    = 0x02;
const uint8_t RSP_MOVED     = 0x81;
const uint8_t RSP_STATUS    = 0x82;

void setup() {
  myServo.attach(SERVO_PIN);
  Serial.begin(115200);
  // Wait for handshake command
  while (true) {
    if (Serial.available() >= 2) {
      uint8_t cmd = Serial.read();
      uint8_t arg = Serial.read(); // Ignored
      if (cmd == CMD_HANDSHAKE) {
        // Reply and break out to normal operation
        Serial.write(RSP_HANDSHAKE);
        Serial.write((uint8_t)0x00);
        break;
      }
    }
  }
  while (Serial.available()) Serial.read(); // Flush leftover bytes after handshake
}

void loop() {
  if (Serial.available() >= 2) {
    uint8_t cmd = Serial.read();
    uint8_t arg = Serial.read();
    uint8_t angle;

    if (cmd == CMD_MOVE) {
      arg = constrain(arg, 0, 180); // Contrain to correct bounds
      myServo.write(arg); // Move servo to input angle
      // Send response
      angle = myServo.read();
      Serial.write(RSP_MOVED);
      Serial.write(angle);
    } else if (cmd == CMD_STATUS) {
      angle = myServo.read();
      Serial.write(RSP_STATUS);
      Serial.write(angle);
    }
  }

}