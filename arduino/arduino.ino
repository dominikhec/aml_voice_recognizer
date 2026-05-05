// file with arduino commends

#include <Adafruit_NeoPixel.h>

#define PIN_STRIP 2

Adafruit_NeoPixel strip(3, PIN_STRIP, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(9600);
  strip.begin();
  strip.show();

  pinMode(2, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == '1') {
      for(int i = 0; i < 3; i++)
        strip.setPixelColor(i, strip.Color(255,0,0));
    }
    if (c == '0') {
      for(int i = 0; i < 3; i++)
        strip.setPixelColor(i, strip.Color(0,0,0));
    }
  }

  strip.show();
}

