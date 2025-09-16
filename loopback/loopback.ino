#include <ESP32Servo.h>

Servo Sprayer;

const int servoPin = 4;
const int BuzzerPin = 5;
const int LED=2;

unsigned long stateStartTime = 0;
unsigned long buzzerTimer = 0;
bool buzzerActive = false;
int buzzerCount = 0;
int maxBuzzerCount = 0;

enum State {
  IDLE,
  MEDIUM_BUZZER,
  HIGH_BUZZER_CONTINUOUS,
  HIGH_BUZZER_PULSED,
  SMELL_ATTACK_SERVO
};

State currentState = IDLE;

void setup() {
  Serial.begin(115200);
  while (!Serial) {;}
  Sprayer.attach(servoPin, 500, 2400);
  Sprayer.write(0);
  Serial.println("Annoyance Machine Ready");
  pinMode(BuzzerPin, OUTPUT);
  pinMode(LED,OUTPUT);
  digitalWrite(BuzzerPin, LOW);
  digitalWrite(LED,LOW);
}

void loop() {
  handleSerialCommands();
  updateStateMachine();
  delay(1);
}

void handleSerialCommands() {
  if (Serial.available()) {
    char incomingMessage = Serial.read();    
    switch(incomingMessage) {
      case 'A':
        Serial.println("SYSTEM_READY");
        break;
      case 'M':
        if (currentState == IDLE) {
          startMediumAlert();
        }
        break;
      case 'H':
        if (currentState == IDLE) {
          startHighAlert();
        }
        break;
      case 'S':
        startSmellAttack();
        break;
      default: 
        break;
    }
  }
}

void startMediumAlert() {
  Serial.println("Annoying with buzzer");
  currentState = MEDIUM_BUZZER;
  stateStartTime = millis();
  buzzerTimer = millis();
  buzzerCount = 0;
  maxBuzzerCount = 3;
  buzzerActive = false;
}

void startHighAlert() {
  Serial.println("Continuous Buzzer Mode Active");
  currentState = HIGH_BUZZER_CONTINUOUS;
  stateStartTime = millis();
  digitalWrite(BuzzerPin, HIGH);
  digitalWrite(LED,HIGH);
}

void startSmellAttack() {
  Serial.println("Commencing Ultimate Attack - OVERRIDE ALL");
  digitalWrite(BuzzerPin, LOW);
  digitalWrite(LED,LOW);
  currentState = SMELL_ATTACK_SERVO;
  stateStartTime = millis();
  Sprayer.write(180);
}

void updateStateMachine() {
  unsigned long currentMillis = millis();
  unsigned long elapsed = currentMillis - stateStartTime;
  unsigned long buzzerElapsed = currentMillis - buzzerTimer;
  
  switch(currentState) {
    case IDLE:
      break;
      
    case MEDIUM_BUZZER:
      if (buzzerCount < maxBuzzerCount) {
        if (!buzzerActive && buzzerElapsed >= 1000) {
          digitalWrite(BuzzerPin, HIGH);
          digitalWrite(LED,HIGH);
          buzzerActive = true;
          buzzerTimer = currentMillis;
        } else if (buzzerActive && buzzerElapsed >= 500) {
          digitalWrite(BuzzerPin, LOW);
          digitalWrite(LED,LOW);
          buzzerActive = false;
          buzzerCount++;
          buzzerTimer = currentMillis;
        }
      } else {
        currentState = IDLE;
        digitalWrite(BuzzerPin, LOW);
        digitalWrite(LED,LOW);
      }
      break;
      
    case HIGH_BUZZER_CONTINUOUS:
      if (elapsed >= 3000) {
        digitalWrite(BuzzerPin, LOW);
        digitalWrite(LED,LOW);
        currentState = HIGH_BUZZER_PULSED;
        stateStartTime = currentMillis;
        buzzerTimer = currentMillis;
        buzzerCount = 0;
        buzzerActive = false;
      }
      break;
      
    case HIGH_BUZZER_PULSED:
      if (buzzerCount < maxBuzzerCount) {
        if (!buzzerActive && buzzerElapsed >= 1000) {
          digitalWrite(BuzzerPin, HIGH);
          digitalWrite(LED,HIGH);
          buzzerActive = true;
          buzzerTimer = currentMillis;
        } else if (buzzerActive && buzzerElapsed >= 500) {
          digitalWrite(BuzzerPin, LOW);
          digitalWrite(LED,LOW);
          buzzerActive = false;
          buzzerCount++;
          buzzerTimer = currentMillis;
        }
      } else {
        currentState = IDLE;
        digitalWrite(BuzzerPin, LOW);
        digitalWrite(LED,LOW);
      }
      break;
      
    case SMELL_ATTACK_SERVO:
      if (elapsed >= 1000) {
        Sprayer.write(0);
        startHighAlert();
      }
      break;
  }
}
