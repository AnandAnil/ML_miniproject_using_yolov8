void setup() {
  // Initialize serial communication at 115200 baud rate
  Serial.begin(115200);
  
  // Wait for serial port to connect (useful for some boards)
  while (!Serial) {
    ; // wait for serial port to connect
  }
  
  // Send startup message
  Serial.println("Arduino Ready - Loopback Test");
}

void loop() {
  // Check if data is available to read
  if (Serial.available()) {
    // Read the incoming character
    char incomingMessage = Serial.read();
    
    // Echo back with more descriptive format
    Serial.print("Received: ");
    Serial.println(incomingMessage);
    
    // Optional: Add specific responses for drowsiness detection testing
    switch(incomingMessage) {
      case 'H':
        Serial.println("HIGH_ALERT_ACTIVATED");
        break;
      case 'M':
        Serial.println("MEDIUM_ALERT_ACTIVATED");
        break;
      case 'N':
        Serial.println("NORMAL_STATUS_SET");
        break;
      default:
        Serial.println("UNKNOWN_COMMAND");
        break;
    }
  }
}