# Hardware pin definitions
SERVO_PINS = {
    'leg1_shoulder': 0,  'leg1_elbow': 1,  'leg1_foot': 2,
    'leg2_shoulder': 3,  'leg2_elbow': 4,  'leg2_foot': 5,
    'leg3_shoulder': 6,  'leg3_elbow': 7,  'leg3_foot': 8,
    'leg4_shoulder': 9,  'leg4_elbow': 10, 'leg4_foot': 11
}

ULTRASONIC_PINS = {
    'trigger': 23,
    'echo': 24
}

I2C_ADDRESSES = {
    'pca9685': 0x40,
    'oled': 0x3C
}