#include <WiFi.h>
//#include <WebSocketsServer.h>
#include <ArduinoJson.h>

// --- WIFI CONFIGURATION ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

WebSocketsServer webSocket = WebSocketsServer(81);

// --- HARDWARE PIN MAPPING ---
struct Motor {
  int stepPin;
  int dirPin;
};

// Based on the architecture document
Motor motors[6] = {
  {13, 12}, // U (Motor 0)
  {14, 27}, // R (Motor 1)
  {26, 25}, // F (Motor 2)
  {33, 32}, // D (Motor 3)
  {15, 2},  // L (Motor 4)
  {4, 5}    // B (Motor 5)
};

const int ENABLE_PIN = 23;
const int stepsPer90 = 400; // 1/8 microstepping
const int stepDelay = 500;  // microseconds

// Mapping character to motor index
int getMotorIdx(char face) {
  switch(face) {
    case 'U': return 0;
    case 'R': return 1;
    case 'F': return 2;
    case 'D': return 3;
    case 'L': return 4;
    case 'B': return 5;
    default: return -1;
  }
}

void executeMove(String move) {
  char face = move[0];
  bool isCCW = move.endsWith("_CCW");
  int motorIdx = getMotorIdx(face);
  
  if (motorIdx == -1) return;
  
  digitalWrite(motors[motorIdx].dirPin, isCCW ? LOW : HIGH);
  
  for (int i = 0; i < stepsPer90; i++) {
    digitalWrite(motors[motorIdx].stepPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(motors[motorIdx].stepPin, LOW);
    delayMicroseconds(stepDelay);
  }
  delay(100); // Settle time
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
  if (type == WStype_TEXT) {
    StaticJsonDocument<2048> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) return;
    
    String msgType = doc["type"];
    if (msgType == "SOLVE") {
      JsonArray moves = doc["moves"];
      int total = moves.size();
      
      digitalWrite(ENABLE_PIN, LOW); // Energize motors
      delay(200);
      
      for (int i = 0; i < total; i++) {
        String move = moves[i];
        executeMove(move);
        
        // Send progress back
        String progress = "{\"type\":\"PROGRESS\",\"current\":" + String(i+1) + ",\"total\":" + String(total) + "}";
        webSocket.broadcastTXT(progress);
      }
      
      webSocket.broadcastTXT("{\"type\":\"COMPLETE\"}");
      delay(1000);
      digitalWrite(ENABLE_PIN, HIGH); // De-energize
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, HIGH); // Disable motors initially
  
  for (int i = 0; i < 6; i++) {
    pinMode(motors[i].stepPin, OUTPUT);
    pinMode(motors[i].dirPin, OUTPUT);
  }
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

void loop() {
  webSocket.loop();
}
