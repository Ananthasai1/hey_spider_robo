import os
from dataclasses import dataclass

@dataclass
class Settings:
    # API Keys
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    
    # Voice Settings
    WAKE_PHRASE: str = "hey spider"
    VOICE_TIMEOUT: int = 5
    
    # Vision Settings
    AUTO_CAPTURE_INTERVAL: int = 30
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # AI Settings
    AI_THINKING_INTERVAL: int = 15
    AI_MODEL: str = "gpt-3.5-turbo"
    
    # Web Settings
    WEB_PORT: int = 5000
    
    # Hardware Settings
    SERVO_FREQUENCY: int = 50
    
    # OLED Settings
    OLED_WIDTH: int = 128
    OLED_HEIGHT: int = 64
    OLED_UPDATE_INTERVAL: float = 0.5

settings = Settings()