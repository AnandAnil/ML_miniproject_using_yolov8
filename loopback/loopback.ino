#include <ESP32Servo.h>

Servo Sprayer;

const int servoPin = 2;
const int BuzzerPin = 4;

void setup() {
  Serial.begin(115200);
  while (!Serial) {;}
  Sprayer.attach(servoPin, 500, 2400);
  Sprayer.write(0);
  Serial.println("Annoyance Machine Ready");
  pinMode(BuzzerPin,OUTPUT);
}

void annoydriver(){
  Serial.println("Annoying with buzzer");
  for (int i = 0; i<=3; i++){
    digitalWrite(BuzzerPin,HIGH);
    delay(500);
    digitalWrite(BuzzerPin,LOW);
    delay(500);
  }
  Serial.println("Annoyed with buzzer");
  delay(200);
}

void attackdriver(){
  Serial.println("Commencing ultimate attack");
  Sprayer.write(180);
  delay(500);
  Sprayer.write(0);
  Serial.println("Ultimate attack successful");
  delay(500);
  annoydriver();
}

void loop() {
  if (Serial.available()) {
    char incomingMessage = Serial.read();
    Serial.print("Received: ");
    Serial.println(incomingMessage);
    switch(incomingMessage) {
      case 'H':
        attackdriver();
        break;
      case 'M':
        annoydriver();
        break;
      default: break;
    }
  }
}