#include <Servo.h>
int SERVO_PIN = 3;

String readString; //String captured from serial port
Servo myServo;  // create servo object to control a servo 
int servoAngle; //value to write to servo

void setup() {
  myServo.attach(SERVO_PIN);
  Serial.begin(11520);
  while (!Serial) {}
  while (!Serial.available()) {}
  readString = Serial.readStringUntil('\n');
  readString.trim();
  if (readString == "MARCO") {
    Serial.println("POLO");
  } else {
    Serial.println("ERROR");
  }
  readString = "";
}

void loop() {

  if (Serial.available()) {
    readString = Serial.readStringUntil('\n');
    readString.trim();

    // MOVE instruction
    if (readString.startsWith("MOVE:")) {
      // Read angle from Serial monitor
      int angle = readString.substring(5).toInt();
      // Write angle to servo
      myServo.write(angle);
      // Send confirmation to Python script
      Serial.print("MOVED:");
      Serial.println(myServo.read());
    } 
    // STATUS instruction
    else if (readString == "STATUS") {
      // Send angle to Python script
      Serial.print("STATUS:");
      Serial.println(myServo.read());
    }
    // Unrecognized instruction
    else {
      Serial.println("ERROR");
    }

    // Reset readString for next instruction
    readString = "";
  }

}