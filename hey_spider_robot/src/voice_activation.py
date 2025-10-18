import threading
import time
from typing import Optional, Callable

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("SpeechRecognition not available - voice commands disabled")
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    print("PyAudio not available - microphone disabled")
    PYAUDIO_AVAILABLE = False

from config.settings import settings
from src.oled_display import OLEDDisplay

class VoiceActivation:
    def __init__(self, command_callback: Callable[[str], None], 
                 oled_display: Optional[OLEDDisplay] = None):
        self.command_callback = command_callback
        self.oled = oled_display
        self.recognizer = None
        self.microphone = None
        self.listening = False
        self.listen_thread = None
        
        # Initialize speech recognition components
        if SPEECH_RECOGNITION_AVAILABLE and PYAUDIO_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                
                # Adjust for ambient noise
                with self.microphone as source:
                    print("Adjusting for ambient noise... Please wait.")
                    self.recognizer.adjust_for_ambient_noise(source, duration=2)
                    print("Voice system initialized")
                    
            except Exception as e:
                print(f"Microphone initialization error: {e}")
                self.recognizer = None
                self.microphone = None
        else:
            print("Voice recognition disabled - required libraries not available")
            
    def start_listening(self):
        """Start the voice recognition thread"""
        if not self.recognizer or not self.microphone:
            print("Voice system not available - using mock voice commands")
            self._start_mock_voice()
            return
            
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        if self.oled:
            self.oled.update_mode("LISTENING")
            
    def _start_mock_voice(self):
        """Start mock voice commands for testing"""
        def mock_commands():
            time.sleep(10)  # Wait 10 seconds
            mock_cmds = ["walk forward", "turn left", "dance", "wave", "take photo"]
            for cmd in mock_cmds:
                if not self.listening:
                    break
                print(f"Mock voice command: {cmd}")
                if self.oled:
                    self.oled.update_command(cmd)
                self.command_callback(cmd)
                time.sleep(15)  # Wait 15 seconds between commands
                
        self.listening = True
        threading.Thread(target=mock_commands, daemon=True).start()
        
    def stop_listening(self):
        """Stop voice recognition"""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2)
            
    def _listen_loop(self):
        """Main listening loop"""
        print("Voice recognition started - say 'Hey Spider' to activate")
        
        while self.listening:
            try:
                self._listen_for_wake_phrase()
                time.sleep(0.1)
            except Exception as e:
                print(f"Listening error: {e}")
                time.sleep(1)
                
    def _listen_for_wake_phrase(self):
        """Listen for the wake phrase and commands"""
        try:
            with self.microphone as source:
                # Short timeout to keep loop responsive
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
            # Recognize speech
            try:
                text = self.recognizer.recognize_google(audio, language='en-US').lower()
            except sr.RequestError:
                # Try offline recognition as fallback
                try:
                    text = self.recognizer.recognize_sphinx(audio).lower()
                except:
                    return
                
            if self.oled:
                self.oled.update_status("Heard: " + text[:15])
                
            # Check for wake phrase
            if settings.WAKE_PHRASE in text:
                # Extract command after wake phrase
                wake_index = text.find(settings.WAKE_PHRASE)
                command = text[wake_index + len(settings.WAKE_PHRASE):].strip()
                
                if command:
                    print(f"Voice command received: {command}")
                    if self.oled:
                        self.oled.update_command(command)
                        self.oled.update_mode("PROCESSING")
                    self.command_callback(command)
                else:
                    print("Wake phrase detected, waiting for command...")
                    if self.oled:
                        self.oled.update_status("Say command...")
                    self._wait_for_command()
                        
        except sr.WaitTimeoutError:
            pass  # Normal timeout, continue listening
        except sr.UnknownValueError:
            pass  # Could not understand audio
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected voice error: {e}")
            time.sleep(2)
            
    def _wait_for_command(self):
        """Wait for command after wake phrase detected"""
        try:
            with self.microphone as source:
                print("Listening for command...")
                audio = self.recognizer.listen(source, timeout=settings.VOICE_TIMEOUT, phrase_time_limit=5)
                
            try:
                command = self.recognizer.recognize_google(audio, language='en-US').lower()
            except sr.RequestError:
                try:
                    command = self.recognizer.recognize_sphinx(audio).lower()
                except:
                    command = ""
                    
            if command:
                print(f"Command: {command}")
                if self.oled:
                    self.oled.update_command(command)
                    self.oled.update_mode("PROCESSING")
                self.command_callback(command)
            else:
                print("No command understood")
                if self.oled:
                    self.oled.update_status("Command not understood")
                
        except sr.WaitTimeoutError:
            print("No command received within timeout")
            if self.oled:
                self.oled.update_status("Command timeout")
        except Exception as e:
            print(f"Command listening error: {e}")
        finally:
            if self.oled:
                self.oled.update_mode("LISTENING")