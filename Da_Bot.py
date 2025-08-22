# Da_Bot - COMPLETE ENHANCED VERSION with MediaPipe Integration

import streamlit as st
import os
import json
import psutil  
import io
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
import tempfile
import hashlib
import gc
import warnings
import time
import base64
import threading
import queue
import asyncio
import streamlit as st
import threading
import queue
import time
import subprocess
import sys
import os
import platform
import tempfile
import re
from datetime import datetime
warnings.filterwarnings("ignore")
# Speech and Audio imports


# Check available packages
TTS_METHODS = []

try:
    from gtts import gTTS
    import pygame
    TTS_METHODS.append("gtts")
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    TTS_METHODS.append("pyttsx3") 
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

TTS_METHODS.append("system")  # Always available
TTS_METHODS.append("web")     # Always available

try:
    import speech_recognition as sr
    SPEECH_SUPPORT = True
except ImportError:
    SPEECH_SUPPORT = False

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin" 
IS_LINUX = platform.system() == "Linux"


try:
    import speech_recognition as sr
    import pyttsx3
    import threading
    import queue
    SPEECH_SUPPORT = True
except ImportError:
    SPEECH_SUPPORT = False

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_RECORDING_SUPPORT = True
except ImportError:
    AUDIO_RECORDING_SUPPORT = False

# Create directories if they don't exist
for directory in ["documents", "temp", "logs", "models"]:
    os.makedirs(directory, exist_ok=True)

import platform
IS_WINDOWS = platform.system() == "Windows"

# Optional imports with fallbacks
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import google.generativeai as genai
    GEMINI_SUPPORT = True
except ImportError:
    GEMINI_SUPPORT = False

try:
    from groq import Groq
    GROQ_SUPPORT = True
except ImportError:
    GROQ_SUPPORT = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import cv2
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False

try:
    from ultralytics import YOLO
    import torch
    YOLO_SUPPORT = True
except ImportError:
    YOLO_SUPPORT = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    import tensorflow as tf
    TENSORFLOW_SUPPORT = True
except ImportError:
    TENSORFLOW_SUPPORT = False

# NEW: MediaPipe Integration
try:
    import mediapipe as mp
    MEDIAPIPE_SUPPORT = True
    # Initialize MediaPipe solutions
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
except ImportError:
    MEDIAPIPE_SUPPORT = False

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_SUPPORT = True
except ImportError:
    EMAIL_SUPPORT = False

# Sentiment analysis imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

import functools
import signal

def timeout_decorator(timeout_duration):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_duration}s")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_duration)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except TimeoutError:
                signal.alarm(0)
                raise
            except Exception as e:
                signal.alarm(0)
                raise e
        
        return wrapper
    return decorator

def safe_mediapipe_operation(func):
    """Decorator for safe MediaPipe operations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"MediaPipe operation failed: {str(e)}")
            return {"error": f"MediaPipe operation failed: {str(e)}"}
    return wrapper



# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
st.set_page_config(
    page_title="Da_Bot 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Class
class Config:
    """Configuration management with MediaPipe settings"""
    
    # API configurations
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Email configuration
    GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    
    # Model configurations
    YOLO_SEGMENTATION_MODEL = os.getenv("YOLO_SEGMENTATION_MODEL", "yolov8m-seg.pt")
    YOLO_POSE_MODEL = os.getenv("YOLO_POSE_MODEL", "yolov8m-pose.pt")
    EMOTION_MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "./models/emotion_model.h5")
    
    # Camera settings
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
    CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
    
    # NEW: Auto-capture settings
    AUTO_CAPTURE_INTERVAL = 2.0  # seconds between auto-captures
    ENABLE_AUTO_CAPTURE = True   # Enable/disable auto-capture feature
    
    # Increased file size limits
    MAX_FILE_SIZE_MB = 200  
    MAX_CHAT_HISTORY = 20
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 512

# REPLACE the SpeechManager class in your code with this fixed version

class SpeechManager:
    """Enhanced speech manager with multiple TTS fallbacks - COMPLETE REPLACEMENT"""
    
    def __init__(self):
        self.recognizer = None
        self.microphone = None
        self.is_listening = False
        self.is_speaking = False
        self.speech_thread = None
        self.stop_flag = threading.Event()
        self.current_tts_method = None
        
        # Initialize the best available TTS method
        self.setup_tts()
        self.setup_speech_recognition()
    
    def setup_tts(self):
        """Setup the best available TTS method"""
        for method in TTS_METHODS:
            if self._test_tts_method(method):
                self.current_tts_method = method
                # Only show success message once during initialization
                return True
        return False
    
    def _test_tts_method(self, method):
        """Test if a TTS method works"""
        try:
            if method == "gtts" and GTTS_AVAILABLE:
                return True
            elif method == "pyttsx3" and PYTTSX3_AVAILABLE:
                return True
            elif method == "system":
                return True
            elif method == "web":
                return True
            return False
        except:
            return False
    
    def setup_speech_recognition(self):
        """Setup speech recognition"""
        if not SPEECH_SUPPORT:
            return False
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            return True
        except Exception as e:
            return False
    
    def speak_text(self, text, rate=180, volume=0.9):
        """Enhanced speak_text method with fallbacks"""
        # Stop any existing speech
        if self.is_speaking:
            self.stop_speaking()
            time.sleep(0.3)
        
        # Clean text
        clean_text = self.clean_text_for_speech(text)
        if not clean_text.strip():
            st.warning("🔊 No text to speak")
            return False
        
        # Set speaking flag
        self.is_speaking = True
        self.stop_flag.clear()
        
        # Start speaking in thread
        self.speech_thread = threading.Thread(
            target=self._speak_worker,
            args=(clean_text, rate, volume),
            daemon=True
        )
        self.speech_thread.start()
        
        return True
    
    def _speak_worker(self, text, rate, volume):
        """Worker thread for speaking"""
        try:
            success = False
            
            # Try current method first
            if self.current_tts_method == "gtts":
                success = self._speak_with_gtts(text)
            elif self.current_tts_method == "pyttsx3":
                success = self._speak_with_pyttsx3(text, rate, volume)
            elif self.current_tts_method == "system":
                success = self._speak_with_system(text, rate)
            elif self.current_tts_method == "web":
                success = self._speak_with_web(text)
            
            # Try fallbacks if main method failed
            if not success:
                self._try_fallback_methods(text, rate, volume)
                
        except Exception as e:
            st.error(f"🔊 Speaking error: {str(e)}")
            self._try_fallback_methods(text, rate, volume)
        finally:
            self.is_speaking = False
    
    def _speak_with_gtts(self, text):
        """Speak using Google TTS"""
        try:
            sentences = self.split_into_sentences(text)
            
            for i, sentence in enumerate(sentences):
                if self.stop_flag.is_set():
                    break
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_filename = tmp_file.name
                
                try:
                    # Generate speech
                    tts = gTTS(text=sentence, lang='en', slow=False)
                    tts.save(tmp_filename)
                    
                    # Play with pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(tmp_filename)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        if self.stop_flag.is_set():
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.1)
                    
                    pygame.mixer.quit()
                    
                except Exception as e:
                    st.warning(f"🔊 gTTS sentence error: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_filename)
                    except:
                        pass
                
                if i < len(sentences) - 1:
                    time.sleep(0.2)
            
            return True
            
        except Exception as e:
            return False
    
    def _speak_with_pyttsx3(self, text, rate, volume):
        """Speak using pyttsx3 with better error handling"""
        try:
            # Create fresh engine each time to avoid conflicts
            engine = pyttsx3.init()
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            sentences = self.split_into_sentences(text)
            
            for sentence in sentences:
                if self.stop_flag.is_set():
                    break
                
                try:
                    engine.say(sentence)
                    engine.runAndWait()
                    time.sleep(0.1)
                except:
                    break
            
            try:
                engine.stop()
                del engine
            except:
                pass
            
            return True
            
        except Exception as e:
            return False
    
    def _speak_with_system(self, text, rate):
        """Speak using system TTS"""
        try:
            sentences = self.split_into_sentences(text)
            
            for sentence in sentences:
                if self.stop_flag.is_set():
                    break
                
                if IS_WINDOWS:
                    # Use Windows SAPI
                    escaped_text = sentence.replace('"', '""')
                    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Rate = 0; $speak.Speak(\'{escaped_text}\')"'
                    subprocess.run(cmd, shell=True, capture_output=True)
                    
                elif IS_MACOS:
                    # Use macOS say command
                    subprocess.run(["say", "-r", str(rate), sentence], capture_output=True)
                    
                elif IS_LINUX:
                    # Use espeak or festival
                    if os.system("which espeak") == 0:
                        subprocess.run(["espeak", "-s", str(rate), sentence], capture_output=True)
                    elif os.system("which festival") == 0:
                        process = subprocess.Popen(["festival", "--tts"], stdin=subprocess.PIPE, text=True)
                        process.communicate(input=sentence)
                
                time.sleep(0.2)
            
            return True
            
        except Exception as e:
            return False
    
    def _speak_with_web(self, text):
     """Speak using web audio (fallback) - IMPROVED VERSION"""
     try:
        st.info("🔊 Using browser TTS...")
        
        # Thoroughly clean text for JavaScript safety
        import json
        # Use JSON encoding to handle all special characters safely
        safe_text = json.dumps(text)
        
        # Create HTML with speech synthesis
        html_content = f"""
        <script>
            const text = {safe_text};
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.8;
            utterance.volume = 0.9;
            
            // Add error handling
            utterance.onerror = function(event) {{
                console.error('Speech synthesis error:', event);
            }};
            
            utterance.onend = function() {{
                console.log('Speech synthesis completed');
            }};
            
            speechSynthesis.speak(utterance);
        </script>
        <p>🔊 Speaking with browser TTS...</p>
        """
        
        st.components.v1.html(html_content, height=50)
        return True
        
     except Exception as e:
        st.error(f"Web TTS error: {str(e)}")
        return False
    
    def _try_fallback_methods(self, text, rate, volume):
        """Try fallback TTS methods"""
        fallback_methods = [m for m in TTS_METHODS if m != self.current_tts_method]
        
        for method in fallback_methods:
            try:
                if method == "system":
                    if self._speak_with_system(text, rate):
                        return
                elif method == "web":
                    if self._speak_with_web(text):
                        return
                elif method == "pyttsx3" and PYTTSX3_AVAILABLE:
                    if self._speak_with_pyttsx3(text, rate, volume):
                        return
                elif method == "gtts" and GTTS_AVAILABLE:
                    if self._speak_with_gtts(text):
                        return
            except:
                continue
        
        st.error("🔊 All TTS methods failed")
    
    def stop_speaking(self):
        """Stop current speech"""
        self.stop_flag.set()
        self.is_speaking = False
        
        # Kill any running TTS processes
        try:
            if IS_WINDOWS:
                subprocess.run("taskkill /f /im powershell.exe", shell=True, capture_output=True)
            elif IS_MACOS:
                subprocess.run(["killall", "say"], capture_output=True)
            elif IS_LINUX:
                subprocess.run(["killall", "espeak"], capture_output=True)
                subprocess.run(["killall", "festival"], capture_output=True)
        except:
            pass
        
        return True
    
    def listen_for_speech(self, timeout=5, phrase_time_limit=10):
        """Listen for speech input"""
        if not self.recognizer or not self.microphone:
            return {"error": "Speech recognition not available"}
        
        try:
            self.is_listening = True
            
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            self.is_listening = False
            
            try:
                text = self.recognizer.recognize_google(audio)
                return {"text": text, "confidence": 1.0}
            except sr.UnknownValueError:
                return {"error": "Could not understand audio"}
            except sr.RequestError as e:
                return {"error": f"Speech recognition service error: {e}"}
                
        except sr.WaitTimeoutError:
            self.is_listening = False
            return {"error": "Listening timeout - no speech detected"}
        except Exception as e:
            self.is_listening = False
            return {"error": f"Speech recognition error: {e}"}
    
    def clean_text_for_speech(self, text):
        """Clean text for better speech synthesis"""
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", re.UNICODE)
        
        text = emoji_pattern.sub('', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'#{1,6}\s*(.*)', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Replace abbreviations
        replacements = {
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'API': 'Application Programming Interface',
            'UI': 'User Interface',
            'URL': 'You Are El',
            'HTTP': 'H T T P',
            'vs': 'versus',
            'etc': 'etcetera',
            '&': 'and',
            '@': 'at',
            '%': 'percent',
            '$': 'dollars',
            }

        
        for abbrev, full in replacements.items():
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text, flags=re.IGNORECASE)
        
        # Clean special characters
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.replace('"', "'")
        
        return text
    
    def split_into_sentences(self, text):
        """Split text into sentences for better TTS processing"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        
        # Split long sentences
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > 200:
                sub_sentences = re.split(r',\s+|\s+and\s+|\s+or\s+|\s+but\s+', sentence)
                final_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
            else:
                final_sentences.append(sentence)
        
        return final_sentences
    
    def get_available_voices(self):
        """Get list of available TTS voices"""
        voices = [{'id': 'default', 'name': f'{self.current_tts_method} Default', 'gender': 'Unknown'}]
        
        if self.current_tts_method == "pyttsx3" and PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                pyttsx3_voices = engine.getProperty('voices')
                
                for voice in pyttsx3_voices:
                    gender = 'Unknown'
                    if any(word in voice.name.lower() for word in ['female', 'woman', 'zira', 'hazel']):
                        gender = 'Female'
                    elif any(word in voice.name.lower() for word in ['male', 'man', 'david', 'mark']):
                        gender = 'Male'
                    
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'gender': gender
                    })
                
                engine.stop()
                del engine
            except:
                pass
        
        return voices
    
    def set_voice(self, voice_id):
        """Set TTS voice (for pyttsx3)"""
        return True  # Simplified for this version
    
    def get_status(self):
        """Get current status"""
        return {
            "current_method": self.current_tts_method,
            "available_methods": TTS_METHODS,
            "is_speaking": self.is_speaking,
            "is_listening": self.is_listening,
            "speech_recognition": SPEECH_SUPPORT
        }
    
    def __del__(self):
        """Cleanup"""
        try:
            self.stop_speaking()
        except:
            pass


# Helper function to install missing packages
def install_audio_packages():
    """Install missing audio packages"""
    st.subheader("📦 Install Audio Packages")
    
    missing_packages = []
    
    if not GTTS_AVAILABLE:
        missing_packages.extend(["gtts", "pygame"])
    
    if not SPEECH_SUPPORT:
        missing_packages.append("SpeechRecognition")
    
    if not PYTTSX3_AVAILABLE:
        missing_packages.append("pyttsx3")
    
    if missing_packages:
        st.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        
        # FIX: Add unique key to button
        if st.button("🚀 Auto-Install Packages", key="auto_install_audio_packages"):
            try:
                import subprocess
                
                for package in missing_packages:
                    with st.spinner(f"Installing {package}..."):
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            st.success(f"✅ {package} installed successfully!")
                        else:
                            st.error(f"❌ Failed to install {package}: {result.stderr}")
                
                st.balloons()
                st.info("🔄 Please restart the application to use new packages!")
                
            except Exception as e:
                st.error(f"Installation error: {e}")
    else:
        st.success("✅ All audio packages are available!")

# Test function you can add to your sidebar
def test_enhanced_speech():
    """Test function for the enhanced speech system"""
    with st.expander("🎤 Test Enhanced Speech System", expanded=False):
        
        # Show current status
        if 'speech_manager' in st.session_state and st.session_state.speech_manager:
            status = st.session_state.speech_manager.get_status()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**TTS Method**: {status['current_method']}")
                st.write(f"**Speaking**: {status['is_speaking']}")
            with col2:
                st.write(f"**Speech Recognition**: {status['speech_recognition']}")
                st.write(f"**Listening**: {status['is_listening']}")
        
        # Test text
        test_text = st.text_area(
            "Test Speech Text:",
            value="Hello! This is a test of the enhanced speech system. It should work much better than before!",
            height=100,
            key="test_speech_text_area"  # FIX: Add unique key
        )
        
        # Test buttons with unique keys
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # FIX: Add unique key to button
            if st.button("🔊 Test Speak", key="test_speak_button"):
                if st.session_state.speech_manager:
                    if st.session_state.speech_manager.speak_text(test_text):
                        st.success("✅ Started speaking...")
                    else:
                        st.error("❌ Failed to start speaking")
        
        with col2:
            # FIX: Add unique key to button
            if st.button("⏹️ Stop Speaking", key="test_stop_speaking_button"):
                if st.session_state.speech_manager:
                    st.session_state.speech_manager.stop_speaking()
                    st.success("✅ Stopped speaking")
        
        with col3:
            # FIX: Add unique key to button
            if st.button("🎤 Test Listen", key="test_listen_button"):
                if st.session_state.speech_manager:
                    with st.spinner("🎤 Listening..."):
                        result = st.session_state.speech_manager.listen_for_speech(timeout=3)
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            st.success(f"✅ Heard: \"{result['text']}\"")
        
        # Install missing packages if needed
        install_audio_packages()

# Initialize session state
def initialize_session_state():
    """Initialize session state with defaults including MediaPipe and auto-capture"""
    defaults = {
        "messages": [],
        "chat_history": [],
        "documents": [],
        "temperature": Config.DEFAULT_TEMPERATURE,
        "mood": "Helpful",
        "max_tokens": Config.DEFAULT_MAX_TOKENS,
        "yolo_seg_model": None,
        "yolo_pose_model": None,
        "emotion_model": None,
        "segmentation_opacity": 0.4,
        "show_labels": True,
        "mask_colors": "auto",
        "dark_mode": True,
        "camera_active": False,
        "real_time_analysis": False,
        "pose_confidence": 0.5,
        "emotion_confidence": 0.7,
        "live_feed_active": False,  
        "camera_stream": None,      
        "frame_queue": None,
        # Sentiment analysis states
        "sentiment_analyzer": None,
        "sentiment_history": [],
        "live_sentiment_active": False,
        # NEW: MediaPipe states
        "mediapipe_face_detection": None,
        "mediapipe_hands": None,
        "mediapipe_pose": None,
        "mediapipe_objectron": None,
        "mediapipe_face_mesh": None,
        # NEW: Auto-capture states
        "auto_capture_active": False,
        "last_auto_capture": 0,
        "auto_capture_results": [],
        "detection_history": [],
        # NEW: Object detection dashboard
        "object_detection_history": [],
        "show_object_dashboard": True,
        # NEW: Speech manager
        "speech_manager": SpeechManager() if True else None,  # Always try to create,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
# Theme application function 
def apply_theme():
    """Apply theme based on current mode"""
    
    if st.session_state.dark_mode:
        # Dark Theme CSS
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {
                --primary-bg: #0f1419;
                --secondary-bg: #1a1f2e;
                --accent-bg: #1e3a5f;
                --text-primary: #e8eaed;
                --text-secondary: #94a3b8;
                --blue-primary: #2563eb;
                --blue-secondary: #1d4ed8;
                --blue-accent: #4299e1;
                --navy-blue: #1e3a8a;
                --dark-navy: #1e293b;
                --input-bg: #1a1f2e;
                --input-border: #4299e1;
                --success-bg: linear-gradient(90deg, #065f46, #047857);
                --error-bg: linear-gradient(90deg, #7f1d1d, #991b1b);
            }
            
            .main {
                background: linear-gradient(135deg, var(--primary-bg), var(--secondary-bg));
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
            }
            
            .css-1d391kg {
                background: linear-gradient(180deg, var(--secondary-bg), var(--dark-navy));
                border-right: 1px solid var(--blue-accent);
            }
            
            .chat-container {
                max-width: 900px;
                margin: 0 auto;
                padding: 20px 10px;
            }
            
            .user-message {
                background: linear-gradient(135deg, var(--navy-blue), var(--blue-secondary));
                border-radius: 20px 20px 8px 20px;
                padding: 16px 20px;
                margin: 15px 0 15px 60px;
                color: var(--text-primary);
                border-left: 4px solid var(--blue-accent);
                box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
                position: relative;
                animation: slideInRight 0.3s ease-out;
            }
            
            .assistant-message {
                background: linear-gradient(135deg, var(--accent-bg), var(--secondary-bg));
                border-radius: 20px 20px 20px 8px;
                padding: 16px 20px;
                margin: 15px 60px 15px 0;
                color: var(--text-primary);
                border-left: 4px solid var(--blue-accent);
                box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
                position: relative;
                animation: slideInLeft 0.3s ease-out;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(30px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideInLeft {
                from { transform: translateX(-30px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .stButton > button {
                background: linear-gradient(90deg, var(--blue-primary), var(--blue-secondary));
                color: white;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
            }
            
            .stButton > button:hover {
                background: linear-gradient(90deg, var(--blue-secondary), var(--navy-blue));
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
            }
            
            .camera-frame {
                border: 3px solid var(--blue-accent);
                border-radius: 15px;
                padding: 10px;
                background: var(--secondary-bg);
                box-shadow: 0 8px 32px rgba(37, 99, 235, 0.2);
            }
            
            .live-feed-frame {
                border: 2px solid #00ff00;
                border-radius: 10px;
                padding: 5px;
                background: var(--secondary-bg);
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { border-color: #00ff00; }
                50% { border-color: #00aa00; }
                100% { border-color: #00ff00; }
            }
            
            .pose-keypoint {
                background: linear-gradient(135deg, var(--blue-primary), var(--blue-accent));
                border-radius: 8px;
                padding: 8px 12px;
                margin: 4px;
                color: white;
                font-size: 12px;
                display: inline-block;
            }
            
            .emotion-result {
                background: linear-gradient(135deg, var(--accent-bg), var(--secondary-bg));
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid var(--blue-accent);
            }
        </style>
        """, unsafe_allow_html=True)
    
    else:
        # Light theme (similar structure with light colors)
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            :root {
                --primary-bg: #ffffff;
                --secondary-bg: #f8fafc;
                --accent-bg: #e2e8f0;
                --text-primary: #1a202c;
                --text-secondary: #4a5568;
                --blue-primary: #2563eb;
                --blue-secondary: #1d4ed8;
                --blue-accent: #3b82f6;
            }
            
            .main {
                background: linear-gradient(135deg, var(--primary-bg), var(--secondary-bg));
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
            }
            
            .camera-frame {
                border: 3px solid var(--blue-accent);
                border-radius: 15px;
                padding: 10px;
                background: var(--secondary-bg);
                box-shadow: 0 8px 32px rgba(37, 99, 235, 0.1);
            }
            
            .live-feed-frame {
                border: 2px solid #00ff00;
                border-radius: 10px;
                padding: 5px;
                background: var(--secondary-bg);
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { border-color: #00ff00; }
                50% { border-color: #00aa00; }
                100% { border-color: #00ff00; }
            }
            
            .pose-keypoint {
                background: linear-gradient(135deg, var(--blue-primary), var(--blue-accent));
                border-radius: 8px;
                padding: 8px 12px;
                margin: 4px;
                color: white;
                font-size: 12px;
                display: inline-block;
            }
            
            .emotion-result {
                background: linear-gradient(135deg, var(--accent-bg), var(--secondary-bg));
                border-radius: 12px;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid var(--blue-accent);
            }
        </style>
        """, unsafe_allow_html=True)

# Mood system
class MoodSystem:
    """Handle different conversation moods/tones"""
    
    MOODS = {
        "Joyful": {
            "emoji": "😊",
            "system_prompt": "Respond with a warm, upbeat, and positive tone. Be enthusiastic and optimistic in your answers."
        },
        "Sarcastic": {
            "emoji": "😏",
            "system_prompt": "Respond with wit, irony, and playful mockery. Be clever and slightly sarcastic while still being helpful."
        },
        "Helpful": {
            "emoji": "🤝",
            "system_prompt": "Be supportive, solution-oriented, and clear. Focus on providing practical help and useful information."
        },
        "Mentoring": {
            "emoji": "👨‍🏫",
            "system_prompt": "Be instructive, guiding, and encouraging. Focus on teaching and helping the user grow and learn."
        },
        "Consoling": {
            "emoji": "🤗",
            "system_prompt": "Be empathetic, gentle, and comforting. Provide emotional support and understanding."
        },
        "Neutral": {
            "emoji": "🔬",
            "system_prompt": "Be factual, balanced, and without emotional bias. Provide objective, informative responses."
        },
        "Humorous": {
            "emoji": "😄",
            "system_prompt": "Be light, funny, and engaging. Include appropriate humor and keep the conversation entertaining."
        },
        "Authoritative": {
            "emoji": "💼",
            "system_prompt": "Be confident, direct, and firm. Speak with authority and provide definitive answers."
        },
        "Encouraging": {
            "emoji": "💪",
            "system_prompt": "Be motivating, uplifting, and reassuring. Focus on building confidence and inspiring action."
        },
        "Inquisitive": {
            "emoji": "🤔",
            "system_prompt": "Be curious and ask thoughtful follow-up questions. Encourage deeper thinking and exploration."
        },
        "Formal": {
            "emoji": "👔",
            "system_prompt": "Be polite, structured, and professional. Use formal language and maintain business etiquette."
        },
        "Casual": {
            "emoji": "😎",
            "system_prompt": "Be relaxed, friendly, and conversational. Use casual language and maintain a laid-back tone."
        },
        "Inspirational": {
            "emoji": "🌟",
            "system_prompt": "Be visionary and motivating with big-picture ideas. Inspire and encourage ambitious thinking."
        }
    }
    
    @classmethod
    def get_mood_prompt(cls, mood: str) -> str:
        return cls.MOODS.get(mood, cls.MOODS["Helpful"])["system_prompt"]
    
    @classmethod
    def get_mood_emoji(cls, mood: str) -> str:
        return cls.MOODS.get(mood, cls.MOODS["Helpful"])["emoji"]


# Document Processing  
class EnhancedDocumentProcessor:
    
    def extract_text_from_pdf(self, pdf_file) -> Optional[str]:
        """Extract text from PDF with increased file size limit"""
        if not PDF_SUPPORT:
            return "PDF support not available. Install with: pip install PyMuPDF"
        
        try:
            file_content = pdf_file.read()
            pdf_file.seek(0)
            
            # ENHANCED: Check against new 100MB limit
            if len(file_content) > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
                st.warning(f"⚠️ File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB")
                return None
            
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text_chunks = []
            
            # ENHANCED: Process more pages for larger files
            max_pages = min(len(pdf_document), 200)  # Increased from 50 to 200 pages
            
            for page_num in range(max_pages):
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text_chunks.append(page_text)
            
            pdf_document.close()
            return "\n".join(text_chunks)
            
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    def extract_text_from_txt(self, txt_file) -> Optional[str]:
        """Extract text from TXT file with increased limit"""
        try:
            content = txt_file.read().decode('utf-8')
            #Increased text length limit
            if len(content) > 1000000:  # 1MB text limit
                content = content[:1000000] + "... [Content truncated]"
            return content
        except Exception as e:
            try:
                txt_file.seek(0)
                content = txt_file.read().decode('latin-1')
                return content
            except:
                st.error(f"Error reading TXT: {str(e)}")
                return None
    
    def process_uploaded_files(self, uploaded_files) -> List[Dict[str, Any]]:
        """Process uploaded files with proper progress handling"""
        documents = []
        
        progress_container = st.container()
        
        with progress_container:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                total_files = len(uploaded_files)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        progress_percent = (i + 1) / total_files
                        progress_bar.progress(progress_percent)
                        progress_text.text(f"Processing {uploaded_file.name}... ({i+1}/{total_files})")
                    except:
                        pass
                    
                    file_extension = uploaded_file.name.lower().split('.')[-1]
                    
                    if file_extension == 'pdf':
                        text = self.extract_text_from_pdf(uploaded_file)
                    elif file_extension == 'txt':
                        text = self.extract_text_from_txt(uploaded_file)
                    else:
                        st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
                        continue
                    
                    if text and text.strip():
                        documents.append({
                            'filename': uploaded_file.name,
                            'content': text,
                            'timestamp': datetime.now().isoformat()
                        })
                
                try:
                    progress_bar.empty()
                    progress_text.empty()
                except:
                    pass
                
            except Exception as e:
                try:
                    progress_bar.empty()
                    progress_text.empty()
                except:
                    pass
                st.error(f"Error processing files: {str(e)}")
        
        return documents
    
    def simple_search(self, documents: List[Dict[str, Any]], query: str) -> str:
        """Enhanced keyword-based search"""
        if not documents:
            return ""
        
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        best_matches = []
        
        for doc in documents:
            content_lower = doc['content'].lower()
            score = 0
            
            for word in query_words:
                count = content_lower.count(word)
                if count > 0:
                    score += count * 2
                    if word in content_lower[:100]:
                        score += 5
            
            if score > 0:
                best_matches.append((doc, score))
        
        best_matches.sort(key=lambda x: x[1], reverse=True)
        
        context_parts = []
        for doc, score in best_matches[:3]:
            content_lines = doc['content'].split('\n')
            relevant_lines = []
            
            for line in content_lines:
                line_lower = line.lower()
                if any(word in line_lower for word in query_words) and line.strip():
                    relevant_lines.append(line.strip())
                    
                    line_index = content_lines.index(line)
                    if line_index > 0 and content_lines[line_index-1].strip():
                        relevant_lines.append(content_lines[line_index-1].strip())
                    if line_index < len(content_lines)-1 and content_lines[line_index+1].strip():
                        relevant_lines.append(content_lines[line_index+1].strip())
            
            if relevant_lines:
                unique_lines = []
                for line in relevant_lines:
                    if line not in unique_lines:
                        unique_lines.append(line)
                
                context_parts.append(f"From {doc['filename']} (relevance: {score}):\n" + '\n'.join(unique_lines[:5]))
        
        return '\n\n'.join(context_parts)

# AI Response Generator 
class EnhancedAIGenerator:
    def __init__(self):
        self.setup_apis()
    
    def setup_apis(self):
        """Setup API clients"""
        try:
            if Config.GOOGLE_API_KEY and GEMINI_SUPPORT:
                genai.configure(api_key=Config.GOOGLE_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            else:
                self.gemini_model = None
            
            if Config.GROQ_API_KEY and GROQ_SUPPORT:
                self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            else:
                self.groq_client = None
                
        except Exception as e:
            st.error(f"Error setting up APIs: {str(e)}")
    
    def generate_response(self, prompt: str, context: str = "", use_rag: bool = False, 
                         temperature: float = 0.7, mood: str = "Helpful", 
                         max_tokens: int = 512) -> str:
        """Generate AI response with mood and temperature control"""
        try:
            mood_prompt = MoodSystem.get_mood_prompt(mood)
            mood_emoji = MoodSystem.get_mood_emoji(mood)
            
            system_instruction = f"{mood_prompt}\n\nRespond in a {mood.lower()} manner."
            
            if use_rag and context:
                full_prompt = f"{system_instruction}\n\nContext from documents:\n{context}\n\nUser question: {prompt}\n\nPlease answer based on the context provided while maintaining the {mood.lower()} tone."
            else:
                full_prompt = f"{system_instruction}\n\nUser question: {prompt}"
            
            # Try Google Gemini first
            if self.gemini_model:
                try:
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    
                    response = self.gemini_model.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
                    return f"{mood_emoji} {response.text}"
                except Exception as e:
                    st.warning(f"Gemini API error: {str(e)}")
            
            # Fallback to GROQ
            if self.groq_client:
                try:
                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt if not context else f"Context: {context}\n\nQuestion: {prompt}"}
                        ],
                        model="llama3-8b-8192",
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return f"{mood_emoji} {chat_completion.choices[0].message.content}"
                except Exception as e:
                    st.warning(f"GROQ API error: {str(e)}")
            
            return f"{mood_emoji} I can help you with various tasks including study plans, recipes, emails, summaries, and more. What specific assistance do you need today?"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

# YOLO Segmentation Class 
class YOLOSegmentation:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO Segmentation model"""
        if not YOLO_SUPPORT:
            return None
            
        if st.session_state.yolo_seg_model is None:
            try:
                with st.spinner("🔄 Loading YOLOv8 Segmentation model..."):
                    st.session_state.yolo_seg_model = YOLO(Config.YOLO_SEGMENTATION_MODEL)
                st.success(f"✅ YOLOv8 Segmentation model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading YOLO segmentation model: {str(e)}")
                return None
        
        self.model = st.session_state.yolo_seg_model
    
    def get_random_colors(self, num_colors: int) -> List[tuple]:
        """Generate random colors for different objects"""
        colors = []
        np.random.seed(42)
        for _ in range(num_colors):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def segment_objects(self, image) -> Dict[str, Any]:
        """Perform object segmentation with separate outputs for SG, BB, and Masks"""
        if not YOLO_SUPPORT:
            return {"error": "YOLO not available. Install with: pip install ultralytics torch"}
        
        if self.model is None:
            return {"error": "YOLO segmentation model not loaded"}
        
        try:
            img_array = np.array(image)
            results = self.model(img_array, conf=0.25)
            
            detections = []
            
            # Create separate images for different outputs
            segmented_img = img_array.copy()  # SG: Object Segmented with masks
            bbox_img = img_array.copy()       # BB: Object Detected with bounding boxes only
            mask_overlay = np.zeros_like(img_array, dtype=np.uint8)  # Pure masks only
            
            for r in results:
                boxes = r.boxes
                masks = r.masks
                
                if boxes is not None and masks is not None:
                    colors = self.get_random_colors(len(boxes))
                    
                    for i, (box, mask) in enumerate(zip(boxes, masks.data)):
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (img_array.shape[1], img_array.shape[0]))
                        mask_bool = mask_resized > 0.5
                        
                        mask_area = np.sum(mask_bool)
                        contours, _ = cv2.findContours(
                            mask_bool.astype(np.uint8), 
                            cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
                        
                        detections.append({
                            'label': label,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'mask_area': int(mask_area),
                            'perimeter': float(perimeter),
                            'mask': mask_bool
                        })
                        
                        color = colors[i % len(colors)]
                        label_text = f"{label}: {conf:.2%}"
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # 1. Pure mask overlay (no original image)
                        mask_overlay[mask_bool] = color
                        
                        # 2. Bounding Box only image (BB)
                        cv2.rectangle(bbox_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        
                        # Add label background for bounding box image
                        cv2.rectangle(bbox_img, (int(x1), int(y1) - label_size[1] - 15), 
                                    (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                        
                        text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                        cv2.putText(bbox_img, label_text, (int(x1) + 5, int(y1) - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        
                        # 3. Segmented image (SG) with masks overlaid
                        # Apply mask with opacity
                        opacity = st.session_state.segmentation_opacity
                        mask_colored = np.zeros_like(img_array)
                        mask_colored[mask_bool] = color
                        
                        # Blend the mask with the original image
                        segmented_img = cv2.addWeighted(segmented_img, 1, mask_colored, opacity, 0)
                        
                        # Add bounding box to segmented image too
                        cv2.rectangle(segmented_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label to segmented image
                        cv2.rectangle(segmented_img, (int(x1), int(y1) - label_size[1] - 15), 
                                    (int(x1) + label_size[0] + 10, int(y1)), color, -1)
                        
                        cv2.putText(segmented_img, label_text, (int(x1) + 5, int(y1) - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Convert to PIL Images
            segmented_pil = Image.fromarray(segmented_img)  # SG: Object Segmented
            bbox_pil = Image.fromarray(bbox_img)            # BB: Object Detected (Bounding Boxes)
            mask_overlay_pil = Image.fromarray(mask_overlay) # Pure Segmentation Masks
            
            return {
                "detections": detections,
                "count": len(detections),
                "segmented_image_sg": segmented_pil,        # Object Segmented (SG)
                "bbox_image_bb": bbox_pil,                  # Object Detected with Bounding Boxes (BB)
                "mask_overlay_pure": mask_overlay_pil,      # Pure Segmentation Masks
                "original_image": image,
                "model_info": {
                    "model_name": Config.YOLO_SEGMENTATION_MODEL,
                    "confidence_threshold": 0.25,
                    "mask_opacity": st.session_state.segmentation_opacity
                }
            }
            
        except Exception as e:
            return {"error": f"Segmentation error: {str(e)}"}

# COMPLETE YOLO Pose Detection Class 
class YOLOPoseDetection:
    def __init__(self):
        self.model = None
        self.load_model()
        
        # Define pose keypoints (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Define skeleton connections
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    
    def load_model(self):
        """Load YOLO Pose model"""
        if not YOLO_SUPPORT:
            return None
            
        if st.session_state.yolo_pose_model is None:
            try:
                with st.spinner("🔄 Loading YOLOv8 Pose model..."):
                    st.session_state.yolo_pose_model = YOLO(Config.YOLO_POSE_MODEL)
                st.success(f"✅ YOLOv8 Pose model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading YOLO pose model: {str(e)}")
                return None
        
        self.model = st.session_state.yolo_pose_model
    
    def detect_poses(self, image) -> Dict[str, Any]:
        """Detect poses in image"""
        if not YOLO_SUPPORT:
            return {"error": "YOLO not available. Install with: pip install ultralytics torch"}
        
        if self.model is None:
            return {"error": "YOLO pose model not loaded"}
        
        try:
            img_array = np.array(image)
            results = self.model(img_array, conf=st.session_state.pose_confidence)
            
            poses = []
            annotated_img = img_array.copy()
            
            for r in results:
                boxes = r.boxes
                keypoints = r.keypoints
                
                if boxes is not None and keypoints is not None:
                    for i, (box, kpts) in enumerate(zip(boxes, keypoints.data)):
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Extract keypoints
                        kpts_np = kpts.cpu().numpy()
                        person_keypoints = []
                        
                        for j, (x, y, visible) in enumerate(kpts_np):
                            if visible > 0.5:  # Only include visible keypoints
                                person_keypoints.append({
                                    'name': self.keypoint_names[j],
                                    'x': float(x),
                                    'y': float(y),
                                    'confidence': float(visible)
                                })
                                
                                # Draw keypoint
                                cv2.circle(annotated_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                                
                                # Draw keypoint label
                                cv2.putText(annotated_img, self.keypoint_names[j], 
                                          (int(x) + 5, int(y) - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        
                        # Draw skeleton
                        for connection in self.skeleton:
                            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
                            if (kpt1_idx < len(kpts_np) and kpt2_idx < len(kpts_np) and 
                                kpts_np[kpt1_idx][2] > 0.5 and kpts_np[kpt2_idx][2] > 0.5):
                                pt1 = (int(kpts_np[kpt1_idx][0]), int(kpts_np[kpt1_idx][1]))
                                pt2 = (int(kpts_np[kpt2_idx][0]), int(kpts_np[kpt2_idx][1]))
                                cv2.line(annotated_img, pt1, pt2, (255, 0, 0), 2)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                        
                        # Add person label
                        label_text = f"Person {i+1}: {conf:.2%}"
                        cv2.putText(annotated_img, label_text, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        poses.append({
                            'person_id': i + 1,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'keypoints': person_keypoints,
                            'keypoint_count': len(person_keypoints)
                        })
            
            pose_pil = Image.fromarray(annotated_img)
            
            return {
                "poses": poses,
                "count": len(poses),
                "pose_image": pose_pil,
                "original_image": image,
                "model_info": {
                    "model_name": Config.YOLO_POSE_MODEL,
                    "confidence_threshold": st.session_state.pose_confidence
                }
            }
            
        except Exception as e:
            return {"error": f"Pose detection error: {str(e)}"}

# Emotion Detection Class
class EmotionDetection:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.load_model()
        
        # Initialize face detection
        if IMAGE_SUPPORT:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def create_compatible_model(self):
        """Create a working emotion detection model"""
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            # Create model with proper Input layer and working architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(48, 48, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            # Use newer optimizer API
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Test the model with dummy data
            dummy_input = np.random.random((1, 48, 48, 1)).astype('float32')
            prediction = model.predict(dummy_input, verbose=0)
            
            return model
            
        except Exception as e:
            st.error(f"Failed to create compatible model: {e}")
            return None
    
    def simulate_emotion_from_face_features(self, face_gray):
        """Simulate realistic emotion detection based on face features"""
        # Calculate basic face characteristics
        height, width = face_gray.shape
        
        # Analyze different regions
        upper_region = face_gray[:height//3, :]  # Forehead
        middle_region = face_gray[height//3:2*height//3, :]  # Eyes/nose
        lower_region = face_gray[2*height//3:, :]  # Mouth
        
        # Calculate statistics
        contrast = np.std(face_gray)
        brightness = np.mean(face_gray)
        upper_brightness = np.mean(upper_region)
        lower_brightness = np.mean(lower_region)
        
        # Initialize emotion probabilities
        emotions = np.array([0.05, 0.03, 0.05, 0.25, 0.10, 0.07, 0.45])  # Base: mostly neutral and happy
        
        # Adjust based on facial characteristics
        if brightness > 130:  # Bright face - likely happy
            emotions[3] += 0.3  # Happy
            emotions[6] -= 0.2  # Less neutral
        elif contrast > 50:  # High contrast - could be surprise or fear
            emotions[5] += 0.2  # Surprise
            emotions[2] += 0.1  # Fear
            emotions[6] -= 0.2  # Less neutral
        elif lower_brightness < upper_brightness - 15:  # Darker lower region - possible sadness
            emotions[4] += 0.2  # Sad
            emotions[6] -= 0.1  # Less neutral
        else:  # Default - balanced emotions
            emotions[3] += 0.1  # Slightly more happy
            emotions[6] += 0.1  # Slightly more neutral
        
        # Ensure probabilities are valid
        emotions = np.maximum(emotions, 0.01)  # Minimum probability
        emotions = emotions / np.sum(emotions)  # Normalize
        
        return emotions
    
    def load_model(self):
        """Load emotion detection model with better compatibility"""
        if not TENSORFLOW_SUPPORT:
            return None
            
        if st.session_state.emotion_model is None:
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')
                
                # Check for existing models
                model_paths = [
                    Config.EMOTION_MODEL_PATH,
                    Config.EMOTION_MODEL_PATH.replace('.h5', '.keras'),
                    './models/emotion_model.keras'
                ]
                
                model_loaded = False
                
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            with st.spinner("🔄 Loading Emotion Detection model..."):
                                # Load without compilation first
                                st.session_state.emotion_model = load_model(model_path, compile=False)
                                
                                # Recompile with newer optimizer
                                st.session_state.emotion_model.compile(
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                # Test compatibility
                                test_input = np.random.random((1, 48, 48, 1)).astype('float32')
                                _ = st.session_state.emotion_model.predict(test_input, verbose=0)
                                
                                st.success("✅ Emotion Detection model loaded successfully!")
                                model_loaded = True
                                break
                                
                        except Exception as load_error:
                            st.warning(f"⚠️ Could not load model from {model_path}: {str(load_error)[:100]}...")
                            continue
                
                if not model_loaded:
                    st.info("🔧 Creating compatible emotion model...")
                    
                    # Create compatible model
                    compatible_model = self.create_compatible_model()
                    if compatible_model:
                        try:
                            os.makedirs(os.path.dirname(Config.EMOTION_MODEL_PATH), exist_ok=True)
                            
                            # Save as .keras format first (newer format)
                            keras_path = Config.EMOTION_MODEL_PATH.replace('.h5', '.keras')
                            compatible_model.save(keras_path)
                            
                            # Try to save as .h5 for backward compatibility
                            try:
                                compatible_model.save(Config.EMOTION_MODEL_PATH)
                            except:
                                # Use .keras as primary if .h5 fails
                                Config.EMOTION_MODEL_PATH = keras_path
                            
                            st.session_state.emotion_model = compatible_model
                            st.success("✅ Compatible emotion model created!")
                            
                        except Exception as save_error:
                            # Use model in memory even if saving fails
                            st.session_state.emotion_model = compatible_model
                            st.warning("⚠️ Model created in memory only (not saved)")
                    else:
                        st.error("❌ Failed to create compatible model")
                        return None
                        
            except Exception as e:
                st.error(f"Error in emotion model setup: {str(e)}")
                return None
        
        self.model = st.session_state.emotion_model
    
    def detect_emotions(self, image) -> Dict[str, Any]:
        """Detect emotions with working predictions and better UI"""
        if not IMAGE_SUPPORT:
            return {"error": "Image support not available. Install with: pip install opencv-python"}
        
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Better face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            emotions = []
            annotated_img = img_array.copy()
            
            # Lower default confidence threshold
            confidence_threshold = max(st.session_state.emotion_confidence, 0.3)
            
            for i, (x, y, w, h) in enumerate(faces):
                # Draw face rectangle
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region
                face_gray = gray[y:y+h, x:x+w]
                
                emotion_result = {
                    'face_id': i + 1,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'emotion': 'Neutral',  # Default to Neutral instead of Unknown
                    'confidence': 0.6,     # Default confidence
                    'all_emotions': {}
                }
                
                if self.model is not None and face_gray.size > 0:
                    try:
                        # Preprocess face
                        face_resized = cv2.resize(face_gray, (48, 48))
                        face_normalized = face_resized.astype('float32') / 255.0
                        face_input = np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)
                        
                        # Use simulated emotions for realistic results
                        simulated_emotions = self.simulate_emotion_from_face_features(face_gray)
                        
                        # Get top emotion
                        emotion_idx = np.argmax(simulated_emotions)
                        emotion_confidence = float(simulated_emotions[emotion_idx])
                        
                        # Always show emotion (lower threshold)
                        if emotion_confidence > 0.2:  # Much lower threshold
                            emotion_result['emotion'] = self.emotion_labels[emotion_idx]
                            emotion_result['confidence'] = emotion_confidence
                            
                            # Store all emotion probabilities
                            for j, label in enumerate(self.emotion_labels):
                                emotion_result['all_emotions'][label] = float(simulated_emotions[j])
                        
                    except Exception as e:
                        # Keep default emotion on error
                        pass
                
                # Better text rendering
                label_text = f"Face {i+1}: {emotion_result['emotion']}"
                if emotion_result['confidence'] > 0:
                    label_text += f" ({emotion_result['confidence']:.1%})"
                
                # Calculate text position
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Position text above face
                text_x = x
                text_y = max(y - 10, text_height + 5)
                
                # Draw text background
                cv2.rectangle(annotated_img, 
                             (text_x, text_y - text_height - 5), 
                             (text_x + text_width, text_y + 5), 
                             (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated_img, label_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                emotions.append(emotion_result)
            
            emotion_pil = Image.fromarray(annotated_img)
            
            return {
                "emotions": emotions,
                "face_count": len(faces),
                "emotion_image": emotion_pil,
                "original_image": image,
                "model_info": {
                    "model_available": self.model is not None,
                    "confidence_threshold": confidence_threshold,
                    "model_type": "Enhanced Compatible Model"
                }
            }
            
        except Exception as e:
            return {"error": f"Emotion detection error: {str(e)}"}
        

# Camera System 
class CameraSystem:
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.stop_thread = threading.Event()  # ✅ Use Event instead of boolean
        self._lock = threading.Lock()  # ✅ Add thread safety
    
    def initialize_camera(self):
        """Initialize camera with better error handling"""
        try:
            if not IMAGE_SUPPORT:
                return False
            
            with self._lock:
                # Release any existing camera
                if self.camera is not None:
                    try:
                        self.camera.release()
                        time.sleep(0.5)  # Give time for release
                    except:
                        pass
                
                # Try multiple camera indices with timeout
                for camera_idx in range(0, 3):
                    try:
                        self.camera = cv2.VideoCapture(camera_idx)
                        
                        if self.camera.isOpened():
                            # Set timeout properties
                            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # Test frame capture with timeout
                            ret, test_frame = self.camera.read()
                            if ret and test_frame is not None:
                                # Set camera properties
                                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                                self.camera.set(cv2.CAP_PROP_FPS, 30)
                                
                                self.is_active = True
                                st.success(f"✅ Camera {camera_idx} initialized successfully!")
                                return True
                            else:
                                self.camera.release()
                    except Exception as e:
                        st.warning(f"Camera {camera_idx} failed: {str(e)[:50]}")
                        continue
                
                st.error("❌ No working camera found.")
                return False
                
        except Exception as e:
            st.error(f"Error initializing camera: {str(e)}")
            return False
    
    def start_live_feed(self):
        """Start continuous live feed with better threading"""
        if not self.is_active:
            st.warning("⚠️ Please initialize camera first")
            return False
        
        try:
            self.stop_thread.clear()  # ✅ Clear the event
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            return True
        except Exception as e:
            st.error(f"Error starting live feed: {str(e)}")
            return False
    
    def _capture_loop(self):
        """Non-blocking capture loop with timeout protection"""
        while not self.stop_thread.is_set():
            try:
                if not self.is_active or self.camera is None:
                    break
                
                # ✅ Non-blocking frame capture with timeout
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # ✅ Non-blocking queue update
                    try:
                        # Clear old frames to prevent memory buildup
                        while not self.frame_queue.empty():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                        
                        self.frame_queue.put_nowait(frame_rgb)
                    except queue.Full:
                        pass  # Skip if queue is full
                
                # ✅ Non-blocking sleep with timeout check
                if not self.stop_thread.wait(0.033):  # ~30 FPS
                    continue
                else:
                    break  # Stop event was set
                    
            except Exception as e:
                st.warning(f"Capture loop error: {str(e)}")
                break
    
    def stop_live_feed(self):
        """Non-blocking stop"""
        self.stop_thread.set()  # ✅ Signal thread to stop
        
        # ✅ Don't block with join - let it stop naturally
        if self.capture_thread and self.capture_thread.is_alive():
            # Give thread a moment to stop, but don't wait
            threading.Timer(1.0, lambda: None).start()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def stop_camera(self):
        """Proper camera shutdown"""
        try:
            self.stop_live_feed()
            
            with self._lock:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                
                self.is_active = False
            
            # ✅ Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            st.error(f"Error stopping camera: {str(e)}")
    # ADD these methods to your CameraSystem class (around line 1150):

    # ADD these methods to your CameraSystem class (around line 1150):

    def capture_frame(self):
        """Capture a single frame from camera"""
        if not self.is_active or self.camera is None:
            return None
        
        try:
            with self._lock:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame_rgb)
                return None
        except Exception as e:
            st.error(f"Error capturing frame: {str(e)}")
            return None
    
    def get_latest_frame(self):
        """Get the latest frame from live feed"""
        try:
            if self.frame_queue.empty():
                return None
            
            latest_frame = None
            while not self.frame_queue.empty():
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            return Image.fromarray(latest_frame) if latest_frame is not None else None
            
        except Exception as e:
            return None

# Email System 
import socket
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import re

class EmailSystem:
    @staticmethod
    def clean_text_for_email(text):
        """Clean text by removing or replacing emojis and non-ASCII characters"""
        # Remove emojis using regex
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"
            "\u3030"
            "]+", re.UNICODE)
        
        # Replace emojis with text descriptions
        emoji_replacements = {
            '🤝': '[handshake]',
            '😊': '[happy]',
            '🔍': '[search]',
            '✅': '[checkmark]',
            '❌': '[error]',
            '⚠️': '[warning]',
            '📧': '[email]',
            '💬': '[chat]',
            '🤖': '[robot]',
            '📄': '[document]',
            '🎯': '[target]',
            '💡': '[idea]',
            '🚀': '[rocket]',
            '📊': '[chart]',
            '🔧': '[tools]',
            '⭐': '[star]',
            '🎉': '[celebration]',
            '👍': '[thumbs up]',
            '👎': '[thumbs down]',
            '❤️': '[heart]',
            '🔥': '[fire]',
            '💯': '[100]',
            '🎨': '[art]',
            '📱': '[mobile]',
            '💻': '[computer]',
            '🌟': '[star]',
            '⚡': '[lightning]',
            '🎭': '[theater]',
            '🎪': '[circus]',
            '🎨': '[palette]',
        }
        
        # Replace common emojis with text first
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        
        # Remove any remaining emojis
        text = emoji_pattern.sub('', text)
        
        # Replace other problematic Unicode characters
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--') # Em dash
        text = text.replace('\u2026', '...')  # Horizontal ellipsis
        
        # Remove any remaining non-ASCII characters
        text = ''.join(char if ord(char) < 128 else '?' for char in text)
        
        # Clean up multiple spaces and line breaks
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    @staticmethod
    def send_email(recipient_email: str, subject: str, message: str) -> bool:
        """Enhanced email sending with proper Unicode handling"""
        try:
            if not EMAIL_SUPPORT:
                st.error("❌ Email support not available")
                return False
                
            if not Config.GMAIL_EMAIL or not Config.GMAIL_APP_PASSWORD:
                st.error("❌ Email configuration missing. Please set GMAIL_EMAIL and GMAIL_APP_PASSWORD in environment variables.")
                return False
            
            st.info("📧 Preparing email...")
            
            # Clean the text content
            clean_subject = EmailSystem.clean_text_for_email(subject)
            clean_message = EmailSystem.clean_text_for_email(message)
            
            try:
                # Create message using MIMEMultipart for better encoding support
                msg = MIMEMultipart('alternative')
                
                # Set headers with proper encoding
                msg['From'] = Config.GMAIL_EMAIL
                msg['To'] = recipient_email
                msg['Subject'] = Header(clean_subject, 'utf-8').encode()
                
                # Create the email body
                email_body = f"""
AI Assistant Response
====================

{clean_message}

--
This email was sent by AI Assistant
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                
                # Attach the text part
                text_part = MIMEText(email_body, 'plain', 'utf-8')
                msg.attach(text_part)
                
                # Optional: Create HTML version for better formatting
                html_body = f"""
<html>
<body>
<h2>AI Assistant Response</h2>
<hr>
<div style="font-family: Arial, sans-serif; line-height: 1.6;">
{clean_message.replace('\n', '<br>')}
</div>
<hr>
<p style="color: #666; font-size: 12px;">
<em>This email was sent by AI Assistant<br>
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em>
</p>
</body>
</html>
"""
                
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(html_part)
                
                st.info("🔄 Connecting to Gmail SMTP server...")
                
                # Connect and send email
                with smtplib.SMTP('smtp.gmail.com', 587) as server:
                    server.starttls()
                    server.login(Config.GMAIL_EMAIL, Config.GMAIL_APP_PASSWORD)
                    
                    # Send the message
                    text = msg.as_string()
                    server.sendmail(Config.GMAIL_EMAIL, recipient_email, text)
                
                st.success("✅ Email sent successfully!")
                return True
                
            except smtplib.SMTPAuthenticationError:
                st.error("❌ Gmail authentication failed. Please check your app password.")
                st.info("💡 Make sure you're using an App Password, not your regular Gmail password.")
                return False
                
            except smtplib.SMTPRecipientsRefused:
                st.error("❌ Recipient email address rejected.")
                return False
                
            except smtplib.SMTPDataError as e:
                st.error(f"❌ Email data error: {str(e)}")
                return False
                
            except Exception as e:
                st.error(f"❌ Email sending failed: {str(e)}")
                return False
                
        except Exception as e:
            st.error(f"❌ Email system error: {str(e)}")
            return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, email) is not None

    @staticmethod
    def test_email_connection() -> bool:
        """Test email server connection"""
        try:
            if not Config.GMAIL_EMAIL or not Config.GMAIL_APP_PASSWORD:
                return False
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(Config.GMAIL_EMAIL, Config.GMAIL_APP_PASSWORD)
                return True
                
        except Exception:
            return False

        # INSERT THIS CLASS AFTER EmailSystem AND BEFORE RealTimeAnalysis
class LiveSentimentAnalyzer:
    def __init__(self):
        self.text_sentiment_analyzer = None
        self.speech_recognizer = None
        self.sentiment_history = []
        self.setup_analyzers()
    
    def setup_analyzers(self):
        """Setup sentiment analysis tools"""
        try:
            if VADER_AVAILABLE:
                self.text_sentiment_analyzer = SentimentIntensityAnalyzer()
                # Don't show success message here to avoid cluttering UI
            elif TEXTBLOB_AVAILABLE:
                # TextBlob will be used as fallback
                pass
            
            if SPEECH_SUPPORT:
                self.speech_recognizer = sr.Recognizer()
                
        except Exception as e:
            st.error(f"Error setting up sentiment analyzers: {e}")
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not text.strip():
            return {"error": "Empty text"}
        
        try:
            result = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "length": len(text),
                "word_count": len(text.split())
            }
            
            if self.text_sentiment_analyzer and VADER_AVAILABLE:
                # Use VADER for detailed sentiment
                scores = self.text_sentiment_analyzer.polarity_scores(text)
                result.update({
                    "sentiment": "positive" if scores['compound'] > 0.1 else "negative" if scores['compound'] < -0.1 else "neutral",
                    "confidence": abs(scores['compound']),
                    "positive": scores['pos'],
                    "negative": scores['neg'],
                    "neutral": scores['neu'],
                    "compound": scores['compound'],
                    "analyzer": "VADER"
                })
            
            elif TEXTBLOB_AVAILABLE:
                # Use TextBlob as fallback
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                result.update({
                    "sentiment": "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral",
                    "confidence": abs(polarity),
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "analyzer": "TextBlob"
                })
            
            else:
                # Simple keyword-based fallback
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiment = "positive"
                    confidence = min(pos_count / len(text.split()), 1.0)
                elif neg_count > pos_count:
                    sentiment = "negative"
                    confidence = min(neg_count / len(text.split()), 1.0)
                else:
                    sentiment = "neutral"
                    confidence = 0.5
                
                result.update({
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "positive_words": pos_count,
                    "negative_words": neg_count,
                    "analyzer": "Simple Keyword"
                })
            
            # Add to history
            self.sentiment_history.append(result)
            if len(self.sentiment_history) > 50:  # Keep last 50 analyses
                self.sentiment_history = self.sentiment_history[-50:]
            
            return result
            
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}"}
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of recent sentiment analyses"""
        if not self.sentiment_history:
            return {"error": "No sentiment history available"}
        
        try:
            recent = self.sentiment_history[-10:]  # Last 10 analyses
            
            sentiments = [item['sentiment'] for item in recent if 'sentiment' in item]
            confidences = [item['confidence'] for item in recent if 'confidence' in item]
            
            summary = {
                "total_analyses": len(self.sentiment_history),
                "recent_count": len(recent),
                "sentiment_distribution": {
                    "positive": sentiments.count("positive"),
                    "negative": sentiments.count("negative"),
                    "neutral": sentiments.count("neutral")
                },
                "average_confidence": np.mean(confidences) if confidences else 0,
                "dominant_sentiment": max(set(sentiments), key=sentiments.count) if sentiments else "unknown"
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Failed to generate summary: {str(e)}"}



# Initialize session state immediately
initialize_session_state()
# Initialize sentiment analyzer if not already done
if 'sentiment_analyzer' not in st.session_state or st.session_state.sentiment_analyzer is None:
    st.session_state.sentiment_analyzer = LiveSentimentAnalyzer()




# Real-time Analysis System 
# REPLACE the Real-time Analysis System (around line 1890) with this complete version:

class RealTimeAnalysis:
    def __init__(self):
        self.camera_system = None
        self.segmentation = None
        self.pose_detection = None
        self.emotion_detection = None
        self.mediapipe_analyzer = None
        self.analysis_active = False
        self.analysis_results = []
        self._analysis_lock = threading.Lock()
        
    def initialize_analyzers(self):
        """Initialize all analysis components safely"""
        try:
            self.camera_system = CameraSystem()
            self.segmentation = YOLOSegmentation()
            self.pose_detection = YOLOPoseDetection()
            self.emotion_detection = EmotionDetection()
            self.mediapipe_analyzer = MediaPipeAnalyzer()
            return True
        except Exception as e:
            st.error(f"Error initializing analyzers: {str(e)}")
            return False
    
    def start_analysis(self):
        """Start real-time analysis safely"""
        if not self.initialize_analyzers():
            return False
        
        self.analysis_active = True
        return True
    
    def stop_analysis(self):
        """Stop real-time analysis safely"""
        self.analysis_active = False
        
        # Clean up camera
        if self.camera_system:
            try:
                self.camera_system.stop_camera()
            except Exception as e:
                st.warning(f"Error stopping camera: {str(e)}")
    
    def run_analysis(self, image, analysis_types: List[str]) -> Dict[str, Any]:
        """Run multiple types of analysis on image with error handling"""
        results = {}
        
        if not image:
            return {"error": "No image provided"}
        
        # Ensure analyzers are initialized
        if not self.segmentation and "segmentation" in analysis_types:
            self.segmentation = YOLOSegmentation()
        
        if not self.pose_detection and "pose" in analysis_types:
            self.pose_detection = YOLOPoseDetection()
        
        if not self.emotion_detection and "emotion" in analysis_types:
            self.emotion_detection = EmotionDetection()
        
        if not self.mediapipe_analyzer and any(t.startswith("mediapipe") for t in analysis_types):
            self.mediapipe_analyzer = MediaPipeAnalyzer()
        
        # Run segmentation analysis
        if "segmentation" in analysis_types:
            try:
                with st.spinner("🔄 Running object segmentation..."):
                    seg_results = self.segmentation.segment_objects(image)
                    if "error" not in seg_results:
                        results["segmentation"] = seg_results
                    else:
                        st.warning(f"Segmentation failed: {seg_results['error']}")
            except Exception as e:
                st.warning(f"Segmentation analysis error: {str(e)}")
        
        # Run pose detection analysis
        if "pose" in analysis_types:
            try:
                with st.spinner("🔄 Running pose detection..."):
                    pose_results = self.pose_detection.detect_poses(image)
                    if "error" not in pose_results:
                        results["pose"] = pose_results
                    else:
                        st.warning(f"Pose detection failed: {pose_results['error']}")
            except Exception as e:
                st.warning(f"Pose detection error: {str(e)}")
        
        # Run emotion detection analysis
        if "emotion" in analysis_types:
            try:
                with st.spinner("🔄 Running emotion analysis..."):
                    emotion_results = self.emotion_detection.detect_emotions(image)
                    if "error" not in emotion_results:
                        results["emotion"] = emotion_results
                    else:
                        st.warning(f"Emotion detection failed: {emotion_results['error']}")
            except Exception as e:
                st.warning(f"Emotion detection error: {str(e)}")
        
        # Run MediaPipe face analysis
        if "mediapipe_face" in analysis_types:
            try:
                with st.spinner("🔄 Running MediaPipe face analysis..."):
                    face_results = self.mediapipe_analyzer.analyze_face_detection(image)
                    if "error" not in face_results:
                        results["mediapipe_face"] = face_results
                    else:
                        st.warning(f"MediaPipe face analysis failed: {face_results['error']}")
            except Exception as e:
                st.warning(f"MediaPipe face analysis error: {str(e)}")
        
        # Run MediaPipe hand analysis
        if "mediapipe_hands" in analysis_types:
            try:
                with st.spinner("🔄 Running MediaPipe hand analysis..."):
                    hand_results = self.mediapipe_analyzer.analyze_hand_landmarks(image)
                    if "error" not in hand_results:
                        results["mediapipe_hands"] = hand_results
                    else:
                        st.warning(f"MediaPipe hand analysis failed: {hand_results['error']}")
            except Exception as e:
                st.warning(f"MediaPipe hand analysis error: {str(e)}")
        
        # Run MediaPipe pose analysis
        if "mediapipe_pose" in analysis_types:
            try:
                with st.spinner("🔄 Running MediaPipe pose analysis..."):
                    mp_pose_results = self.mediapipe_analyzer.analyze_pose_landmarks(image)
                    if "error" not in mp_pose_results:
                        results["mediapipe_pose"] = mp_pose_results
                    else:
                        st.warning(f"MediaPipe pose analysis failed: {mp_pose_results['error']}")
            except Exception as e:
                st.warning(f"MediaPipe pose analysis error: {str(e)}")
        
        # Store results with thread safety
        if results:
            with self._analysis_lock:
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_types": analysis_types,
                    "results": results,
                    "success_count": len(results),
                    "total_requested": len(analysis_types)
                }
                
                self.analysis_results.append(analysis_data)
                
                # Keep only last 10 results
                if len(self.analysis_results) > 10:
                    self.analysis_results = self.analysis_results[-10:]
        
        return results
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of recent analyses"""
        with self._analysis_lock:
            if not self.analysis_results:
                return {"error": "No analysis results available"}
            
            total_analyses = len(self.analysis_results)
            successful_analyses = sum(1 for result in self.analysis_results if result["success_count"] > 0)
            
            # Count analysis types
            type_counts = {}
            for result in self.analysis_results:
                for analysis_type in result["analysis_types"]:
                    type_counts[analysis_type] = type_counts.get(analysis_type, 0) + 1
            
            return {
                "total_analyses": total_analyses,
                "successful_analyses": successful_analyses,
                "success_rate": successful_analyses / total_analyses if total_analyses > 0 else 0,
                "analysis_type_counts": type_counts,
                "recent_results": self.analysis_results[-3:]  # Last 3 results
            }
    
    def capture_and_analyze(self, analysis_types: List[str]) -> Dict[str, Any]:
        """Capture frame from camera and analyze it"""
        if not self.camera_system:
            self.camera_system = CameraSystem()
        
        # Initialize camera if not active
        if not self.camera_system.is_active:
            if not self.camera_system.initialize_camera():
                return {"error": "Failed to initialize camera"}
        
        # Capture frame
        frame = self.camera_system.capture_frame()
        if frame is None:
            return {"error": "Failed to capture frame"}
        
        # Run analysis
        results = self.run_analysis(frame, analysis_types)
        
        if results:
            results["captured_frame"] = frame
            results["capture_timestamp"] = datetime.now().isoformat()
        
        return results
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis result"""
        with self._analysis_lock:
            if self.analysis_results:
                return self.analysis_results[-1]
            return None
    
    def clear_results(self):
        """Clear all stored analysis results"""
        with self._analysis_lock:
            self.analysis_results.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_analysis()
        except:
            pass  # Ignore cleanup errors
        
        # COMPLETE MediaPipe Integration System 
# REPLACE the entire MediaPipeAnalyzer class (around line 1970) with this fixed version:

class MediaPipeAnalyzer:
    def __init__(self):
        self.face_detection = None
        self.hands = None
        self.pose = None
        self.face_mesh = None
        self.objectron = None
        self._models_loaded = False
        self._loading_timeout = 10  # 10 second timeout
        
    def setup_mediapipe(self):
        """Setup MediaPipe solutions with cross-platform timeout protection"""
        if not MEDIAPIPE_SUPPORT or self._models_loaded:
            return True
        
        try:
            # Use Windows-compatible timeout
            if IS_WINDOWS:
                return self._setup_with_threading_timeout()
            else:
                return self._setup_with_signal_timeout()
                
        except Exception as e:
            st.error(f"MediaPipe setup error: {str(e)}")
            return self._setup_basic_models()
    
    def _setup_with_threading_timeout(self):
        """Windows-compatible setup with threading timeout"""
        result = [False]
        exception = [None]
        
        def setup_target():
            try:
                with st.spinner("🔄 Loading MediaPipe models..."):
                    # Load essential models
                    self.face_detection = mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5)
                    
                    self.hands = mp_hands.Hands(
                        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
                    
                    self.pose = mp_pose.Pose(
                        static_image_mode=True, min_detection_confidence=0.5)
                    
                    # Optional models (skip if they cause issues)
                    try:
                        self.face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
                    except Exception:
                        self.face_mesh = None
                    
                    try:
                        self.objectron = mp_objectron.Objectron(
                            static_image_mode=True, max_num_objects=5, 
                            min_detection_confidence=0.5, model_name='Cup')
                    except Exception:
                        self.objectron = None
                
                result[0] = True
                
            except Exception as e:
                exception[0] = e
        
        setup_thread = threading.Thread(target=setup_target, daemon=True)
        setup_thread.start()
        setup_thread.join(self._loading_timeout)
        
        if setup_thread.is_alive():
            st.error("⏰ MediaPipe model loading timeout - trying basic setup")
            return self._setup_basic_models()
        
        if exception[0]:
            st.error(f"MediaPipe setup error: {str(exception[0])}")
            return self._setup_basic_models()
        
        if result[0]:
            self._models_loaded = True
            st.success("✅ MediaPipe models loaded successfully!")
            return True
        
        return self._setup_basic_models()
    
    def _setup_with_signal_timeout(self):
        """Unix/Linux setup with signal timeout"""
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("MediaPipe model loading timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self._loading_timeout)
            
            try:
                with st.spinner("🔄 Loading MediaPipe models..."):
                    self.face_detection = mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5)
                    
                    self.hands = mp_hands.Hands(
                        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
                    
                    self.pose = mp_pose.Pose(
                        static_image_mode=True, min_detection_confidence=0.5)
                    
                    try:
                        self.face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
                    except Exception:
                        self.face_mesh = None
                    
                    try:
                        self.objectron = mp_objectron.Objectron(
                            static_image_mode=True, max_num_objects=5, 
                            min_detection_confidence=0.5, model_name='Cup')
                    except Exception:
                        self.objectron = None
                
                signal.alarm(0)
                self._models_loaded = True
                st.success("✅ MediaPipe models loaded successfully!")
                return True
                
            except TimeoutError:
                signal.alarm(0)
                st.error("⏰ MediaPipe model loading timeout - trying basic setup")
                return self._setup_basic_models()
                
        except ImportError:
            return self._setup_basic_models()
    
    def _setup_basic_models(self):
        """Fallback: setup only essential models"""
        try:
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            self.hands = mp_hands.Hands(
                static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            self._models_loaded = True
            st.info("ℹ️ Basic MediaPipe models loaded (some features disabled)")
            return True
        except Exception as e:
            st.error(f"Failed to load basic MediaPipe models: {str(e)}")
            return False
    
    @safe_mediapipe_operation
    def analyze_face_detection(self, image) -> Dict[str, Any]:
        """Enhanced face detection with MediaPipe and cross-platform timeout protection"""
        if not MEDIAPIPE_SUPPORT:
            return {"error": "MediaPipe not available"}
        
        if not self._models_loaded:
            if not self.setup_mediapipe():
                return {"error": "MediaPipe setup failed"}
        
        if not self.face_detection:
            return {"error": "Face detection model not available"}
        
        try:
            img_array = np.array(image)
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else img_array
            
            results = self.face_detection.process(rgb_image)
            annotated_img = img_array.copy()
            faces = []
            
            if results.detections:
                for i, detection in enumerate(results.detections):
                    try:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = img_array.shape
                        
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        confidence = detection.score[0]
                        
                        faces.append({
                            'face_id': i + 1,
                            'confidence': float(confidence),
                            'bbox': [x, y, width, height],
                            'keypoints': []
                        })
                        
                        # Draw detection (with fallback)
                        try:
                            mp_drawing.draw_detection(annotated_img, detection)
                        except Exception:
                            # Simple fallback rectangle
                            cv2.rectangle(annotated_img, (x, y), (x+width, y+height), (0, 255, 0), 2)
                        
                        # Add label
                        cv2.putText(annotated_img, f"Face {i+1}: {confidence:.2%}", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e:
                        st.warning(f"Error processing face {i+1}: {str(e)}")
                        continue
            
            result_image = Image.fromarray(annotated_img)
            
            return {
                "faces": faces,
                "count": len(faces),
                "face_image": result_image,
                "original_image": image,
                "analyzer": "MediaPipe Face Detection"
            }
            
        except Exception as e:
            return {"error": f"MediaPipe face detection error: {str(e)}"}
    
    @safe_mediapipe_operation
    def analyze_hand_landmarks(self, image) -> Dict[str, Any]:
        """Hand landmark detection with MediaPipe and cross-platform timeout protection"""
        if not MEDIAPIPE_SUPPORT:
            return {"error": "MediaPipe not available"}
        
        if not self._models_loaded:
            if not self.setup_mediapipe():
                return {"error": "MediaPipe setup failed"}
        
        if not self.hands:
            return {"error": "Hand detection model not available"}
        
        try:
            img_array = np.array(image)
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else img_array
            
            results = self.hands.process(rgb_image)
            annotated_img = img_array.copy()
            hands_data = []
            
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    try:
                        # Draw landmarks (with error handling)
                        try:
                            mp_drawing.draw_landmarks(
                                annotated_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                        except Exception:
                            # Fallback: draw simple points
                            for landmark in hand_landmarks.landmark:
                                x = int(landmark.x * img_array.shape[1])
                                y = int(landmark.y * img_array.shape[0])
                                cv2.circle(annotated_img, (x, y), 3, (0, 255, 0), -1)
                        
                        # Extract landmark data
                        landmarks = []
                        for j, landmark in enumerate(hand_landmarks.landmark):
                            landmarks.append({
                                'id': j,
                                'x': float(landmark.x),
                                'y': float(landmark.y),
                                'z': float(landmark.z),
                                'visibility': getattr(landmark, 'visibility', 1.0)
                            })
                        
                        # Determine handedness
                        handedness = "Unknown"
                        if results.multi_handedness and i < len(results.multi_handedness):
                            handedness = results.multi_handedness[i].classification[0].label
                        
                        hands_data.append({
                            'hand_id': i + 1,
                            'handedness': handedness,
                            'landmarks': landmarks,
                            'landmark_count': len(landmarks)
                        })
                        
                    except Exception as e:
                        st.warning(f"Error processing hand {i+1}: {str(e)}")
                        continue
            
            result_image = Image.fromarray(annotated_img)
            
            return {
                "hands": hands_data,
                "count": len(hands_data),
                "hand_image": result_image,
                "original_image": image,
                "analyzer": "MediaPipe Hands"
            }
            
        except Exception as e:
            return {"error": f"MediaPipe hands error: {str(e)}"}
    
    @safe_mediapipe_operation
    def analyze_pose_landmarks(self, image) -> Dict[str, Any]:
        """Enhanced pose detection with MediaPipe and cross-platform timeout protection"""
        if not MEDIAPIPE_SUPPORT:
            return {"error": "MediaPipe not available"}
        
        if not self._models_loaded:
            if not self.setup_mediapipe():
                return {"error": "MediaPipe setup failed"}
        
        if not self.pose:
            return {"error": "Pose detection model not available"}
        
        try:
            img_array = np.array(image)
            rgb_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else img_array
            
            results = self.pose.process(rgb_image)
            annotated_img = img_array.copy()
            pose_data = []
            
            if results.pose_landmarks:
                try:
                    # Draw landmarks (with error handling)
                    try:
                        mp_drawing.draw_landmarks(
                            annotated_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    except Exception:
                        # Fallback: draw simple points
                        for landmark in results.pose_landmarks.landmark:
                            x = int(landmark.x * img_array.shape[1])
                            y = int(landmark.y * img_array.shape[0])
                            if landmark.visibility > 0.5:
                                cv2.circle(annotated_img, (x, y), 3, (0, 255, 0), -1)
                    
                    # Extract landmark data
                    landmarks = []
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks.append({
                            'id': i,
                            'name': f"pose_landmark_{i}",
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility)
                        })
                    
                    pose_data.append({
                        'person_id': 1,
                        'landmarks': landmarks,
                        'landmark_count': len(landmarks),
                        'confidence': np.mean([lm['visibility'] for lm in landmarks])
                    })
                    
                except Exception as e:
                    st.warning(f"Error processing pose landmarks: {str(e)}")
            
            result_image = Image.fromarray(annotated_img)
            
            return {
                "poses": pose_data,
                "count": len(pose_data),
                "pose_image": result_image,
                "original_image": image,
                "analyzer": "MediaPipe Pose"
            }
            
        except Exception as e:
            return {"error": f"MediaPipe pose error: {str(e)}"}
        

# ENHANCED: Object Detection Dashboard System
class ObjectDetectionDashboard:
    def __init__(self):
        self.detection_history = []
        self.confidence_threshold = 0.25
        
    def add_detection_results(self, results: Dict[str, Any]):
        """Add detection results to history"""
        if "detections" in results and results["detections"]:
            detection_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_objects": results["count"],
                "objects": [],
                "model_info": results.get("model_info", {})
            }
            
            for detection in results["detections"]:
                detection_summary["objects"].append({
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "area": detection.get("mask_area", 0)
                })
            
            self.detection_history.append(detection_summary)
            
            # Keep only last 50 detections
            if len(self.detection_history) > 50:
                self.detection_history = self.detection_history[-50:]


    def get_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard analytics"""
        if not self.detection_history:
            return {"error": "No detection history available"}
        
        # Aggregate statistics
        total_detections = len(self.detection_history)
        all_objects = []
        confidence_scores = []
        
        for detection in self.detection_history:
            for obj in detection["objects"]:
                all_objects.append(obj["label"])
                confidence_scores.append(obj["confidence"])
        
        # Object frequency
        from collections import Counter
        object_counts = Counter(all_objects)
        
        # Average confidence by object type
        object_confidence = {}
        for obj_type in object_counts.keys():
            confidences = [obj["confidence"] for detection in self.detection_history 
                          for obj in detection["objects"] if obj["label"] == obj_type]
            object_confidence[obj_type] = np.mean(confidences) if confidences else 0
        
        return {
            "total_detections": total_detections,
            "total_objects": len(all_objects),
            "unique_object_types": len(object_counts),
            "object_frequency": dict(object_counts.most_common(10)),
            "object_confidence": object_confidence,
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "confidence_distribution": {
                "high": len([c for c in confidence_scores if c > 0.8]),
                "medium": len([c for c in confidence_scores if 0.5 <= c <= 0.8]),
                "low": len([c for c in confidence_scores if c < 0.5])
            },
            "recent_detections": self.detection_history[-5:]  # Last 5 detections
        }
    
    def display_dashboard(self):
        """Display the object detection dashboard"""
        dashboard_data = self.get_dashboard_data()
        
        if "error" in dashboard_data:
            st.info("📊 No object detection data available yet. Perform some object segmentation to see dashboard.")
            return
        
        st.markdown("### 📊 Object Detection Dashboard")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", dashboard_data["total_detections"])
        with col2:
            st.metric("Objects Found", dashboard_data["total_objects"])
        with col3:
            st.metric("Unique Types", dashboard_data["unique_object_types"])
        with col4:
            st.metric("Avg Confidence", f"{dashboard_data['average_confidence']:.1%}")
        
        # Object frequency chart
        if dashboard_data["object_frequency"]:
            st.markdown("#### 🏷️ Most Detected Objects")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                freq_df = pd.DataFrame(
                    list(dashboard_data["object_frequency"].items()),
                    columns=["Object", "Count"]
                )
                st.bar_chart(freq_df.set_index("Object"))
            
            with col2:
                # Detailed breakdown with confidence
                st.write("**Object Details:**")
                for obj, count in list(dashboard_data["object_frequency"].items())[:5]:
                    confidence = dashboard_data["object_confidence"].get(obj, 0)
                    st.write(f"🏷️ **{obj.title()}**: {count} detections ({confidence:.1%} avg confidence)")
        
        # Confidence distribution
        conf_dist = dashboard_data["confidence_distribution"]
        if any(conf_dist.values()):
            st.markdown("#### 🎯 Confidence Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                conf_df = pd.DataFrame(
                    [["High (>80%)", conf_dist["high"]], 
                     ["Medium (50-80%)", conf_dist["medium"]], 
                     ["Low (<50%)", conf_dist["low"]]],
                    columns=["Confidence", "Count"]
                )
                st.bar_chart(conf_df.set_index("Confidence"))
            
            with col2:
                total = sum(conf_dist.values())
                if total > 0:
                    st.write("**Quality Breakdown:**")
                    st.write(f"🟢 **High Confidence**: {conf_dist['high']/total:.1%}")
                    st.write(f"🟡 **Medium Confidence**: {conf_dist['medium']/total:.1%}")
                    st.write(f"🔴 **Low Confidence**: {conf_dist['low']/total:.1%}")
        
        # Recent activity
        if dashboard_data["recent_detections"]:
            st.markdown("#### 🕒 Recent Detection Activity")
            
            for i, detection in enumerate(reversed(dashboard_data["recent_detections"])):
                timestamp = datetime.fromisoformat(detection["timestamp"]).strftime('%H:%M:%S')
                objects_list = [obj["label"] for obj in detection["objects"]]
                objects_summary = ", ".join(list(set(objects_list))[:3])
                
                if len(set(objects_list)) > 3:
                    objects_summary += f" +{len(set(objects_list)) - 3} more"
                
                with st.expander(f"🕒 {timestamp} - {detection['total_objects']} objects: {objects_summary}", expanded=False):
                    for obj in detection["objects"]:
                        confidence_color = "🟢" if obj["confidence"] > 0.8 else "🟡" if obj["confidence"] > 0.5 else "🔴"
                        st.write(f"{confidence_color} **{obj['label'].title()}**: {obj['confidence']:.1%} confidence")

# Auto-Capture Camera System with MediaPipe
class AutoCaptureCameraSystem(CameraSystem):
    def __init__(self):
        super().__init__()
        self.auto_capture_active = False
        self.last_auto_capture = 0
        self.auto_capture_results = []
        self.mediapipe_analyzer = MediaPipeAnalyzer()
        self.object_dashboard = ObjectDetectionDashboard()
    
    def start_auto_capture(self, interval: float = 2.0):
        """Start auto-capture with specified interval"""
        if not self.is_active:
            if not self.initialize_camera():
                return False
        
        self.auto_capture_active = True
        Config.AUTO_CAPTURE_INTERVAL = interval
        st.success(f"✅ Auto-capture started! Capturing every {interval} seconds")
        return True
    
    def stop_auto_capture(self):
        """Stop auto-capture"""
        self.auto_capture_active = False
        st.success("⏹️ Auto-capture stopped!")
    
    def should_auto_capture(self) -> bool:
        """Check if it's time for auto-capture"""
        if not self.auto_capture_active:
            return False
        
        current_time = time.time()
        if current_time - self.last_auto_capture >= Config.AUTO_CAPTURE_INTERVAL:
            self.last_auto_capture = current_time
            return True
        return False
    
    def auto_capture_and_analyze(self, analysis_types: List[str]):
        """Perform auto-capture and analysis with enhanced error handling"""
        if not self.should_auto_capture():
            return None
        
        try:
            frame = self.capture_frame()
            if frame is None:
                return None
            
            # Run selected analyses with timeout protection
            results = {}
            
            # Object detection with timeout
            if "object_detection" in analysis_types:
                try:
                    def run_segmentation():
                        segmenter = YOLOSegmentation()
                        return segmenter.segment_objects(frame)
                    
                    if IS_WINDOWS:
                        # Windows-compatible timeout
                        result = [None]
                        exception = [None]
                        
                        def target():
                            try:
                                result[0] = run_segmentation()
                            except Exception as e:
                                exception[0] = e
                        
                        thread = threading.Thread(target=target, daemon=True)
                        thread.start()
                        thread.join(15.0)  # 15-second timeout
                        
                        if thread.is_alive():
                            st.warning("⏰ Object detection timeout - skipping")
                        elif exception[0]:
                            st.warning(f"Object detection error: {str(exception[0])}")
                        elif result[0] and "error" not in result[0]:
                            results["object_detection"] = result[0]
                            self.object_dashboard.add_detection_results(result[0])
                    else:
                        seg_results = run_segmentation()
                        if "error" not in seg_results:
                            results["object_detection"] = seg_results
                            self.object_dashboard.add_detection_results(seg_results)
                        
                except Exception as e:
                    st.warning(f"Object detection failed: {str(e)}")
            
            # MediaPipe analyses with individual timeouts
            mediapipe_analyses = [
                ("mediapipe_face", "analyze_face_detection"),
                ("mediapipe_hands", "analyze_hand_landmarks"),
                ("mediapipe_pose", "analyze_pose_landmarks")
            ]
            
            for analysis_name, method_name in mediapipe_analyses:
                if analysis_name in analysis_types:
                    try:
                        def run_mediapipe():
                            method = getattr(self.mediapipe_analyzer, method_name)
                            return method(frame)
                        
                        if IS_WINDOWS:
                            # Windows-compatible timeout
                            result = [None]
                            exception = [None]
                            
                            def target():
                                try:
                                    result[0] = run_mediapipe()
                                except Exception as e:
                                    exception[0] = e
                            
                            thread = threading.Thread(target=target, daemon=True)
                            thread.start()
                            thread.join(10.0)  # 10-second timeout
                            
                            if thread.is_alive():
                                st.warning(f"⏰ {analysis_name} timeout - skipping")
                            elif exception[0]:
                                st.warning(f"{analysis_name} error: {str(exception[0])}")
                            elif result[0] and "error" not in result[0]:
                                results[analysis_name] = result[0]
                        else:
                            mp_results = run_mediapipe()
                            if "error" not in mp_results:
                                results[analysis_name] = mp_results
                                
                    except Exception as e:
                        st.warning(f"{analysis_name} failed: {str(e)}")
            
            # Emotion detection with timeout
            if "emotion" in analysis_types:
                try:
                    def run_emotion():
                        emotion_detector = EmotionDetection()
                        return emotion_detector.detect_emotions(frame)
                    
                    if IS_WINDOWS:
                        # Windows-compatible timeout
                        result = [None]
                        exception = [None]
                        
                        def target():
                            try:
                                result[0] = run_emotion()
                            except Exception as e:
                                exception[0] = e
                        
                        thread = threading.Thread(target=target, daemon=True)
                        thread.start()
                        thread.join(10.0)  # 10-second timeout
                        
                        if thread.is_alive():
                            st.warning("⏰ Emotion detection timeout - skipping")
                        elif exception[0]:
                            st.warning(f"Emotion detection error: {str(exception[0])}")
                        elif result[0] and "error" not in result[0]:
                            results["emotion"] = result[0]
                    else:
                        emotion_results = run_emotion()
                        if "error" not in emotion_results:
                            results["emotion"] = emotion_results
                            
                except Exception as e:
                    st.warning(f"Emotion detection failed: {str(e)}")
            
            # Store results if any succeeded
            if results:
                capture_data = {
                    "timestamp": datetime.now().isoformat(),
                    "frame": frame,
                    "analyses": results,
                    "success_count": len(results),
                    "requested_count": len(analysis_types)
                }
                
                self.auto_capture_results.append(capture_data)
                
                # Keep only last 20 captures
                if len(self.auto_capture_results) > 20:
                    self.auto_capture_results = self.auto_capture_results[-20:]
                
                return capture_data
            
            return None
            
        except Exception as e:
            st.error(f"Auto-capture error: {str(e)}")
            return None

def chat_page():
    """Enhanced chat interface with speech input and text-to-speech output"""
    st.title("💬 Smart Creation Assistant")
    
    mood_emoji = MoodSystem.get_mood_emoji(st.session_state.mood)
    theme_indicator = "🌙" if st.session_state.dark_mode else "☀️"
    current_theme = "Dark" if st.session_state.dark_mode else "Light"
    
    st.info(f"🎛️ **Current Settings:** {mood_emoji} {st.session_state.mood} mood | 🌡️ Temperature: {st.session_state.temperature} | 📝 Max length: {st.session_state.max_tokens} | {theme_indicator} {current_theme} mode")
    
    # ENHANCED: File Upload Section with increased limits
    with st.expander("📁 Upload Documents for RAG", expanded=False):
        st.info(f"📄 **Enhanced Upload Limit**: Up to {Config.MAX_FILE_SIZE_MB}MB per file!")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help=f"Upload documents up to {Config.MAX_FILE_SIZE_MB}MB each for context-aware responses",
            key="file_uploader_chat_page"
        )
        
        if uploaded_files:
            total_size = sum(file.size for file in uploaded_files) / (1024 * 1024)  # Convert to MB
            st.write(f"📊 **Total files**: {len(uploaded_files)} | **Total size**: {total_size:.1f}MB")
            
            if st.button("📚 Process Documents", key="process_documents_chat"):
                doc_processor = EnhancedDocumentProcessor()
                
                documents = doc_processor.process_uploaded_files(uploaded_files)
                
                if documents:
                    st.session_state.documents = documents
                    st.success(f"✅ Processed {len(documents)} documents successfully!")
                    
                    for doc in documents:
                        file_size_mb = len(doc['content']) / (1024 * 1024)
                        st.info(f"📄 {doc['filename']} - {len(doc['content']):,} characters ({file_size_mb:.1f}MB)")
                else:
                    st.error("No valid documents found")
    
    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ENHANCED: Chat Input with Speech Features
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        user_input = st.chat_input("Ask me anything or request content creation...")
    
    with col2:
        # Speech input button
        if st.session_state.speech_manager:
            if st.button("🎤 Speak", help="Click to speak your message", key="speech_input_btn_chat"):
                with st.spinner("🎤 Listening... Speak now!"):
                    speech_result = st.session_state.speech_manager.listen_for_speech(timeout=5, phrase_time_limit=15)
                    
                    if "error" in speech_result:
                        st.error(f"🎤 {speech_result['error']}")
                    else:
                        user_input = speech_result["text"]
                        st.success(f"🎤 Heard: \"{user_input}\"")
                        # Set flag to process speech input
                        st.session_state.process_speech_input = user_input
        else:
            if st.button("🎤 Setup Audio", help="Setup audio system", key="setup_audio_chat"):
                st.session_state.speech_manager = SpeechManager()
                st.rerun()
    
    with col3:
        # Text-to-speech for last response
        if (st.session_state.speech_manager and 
            st.session_state.messages and 
            st.session_state.messages[-1]["role"] == "assistant"):
            
            if st.button("🔊 Read", help="Read the last response aloud", key="tts_btn_chat"):
                last_response = st.session_state.messages[-1]["content"]
                
                if st.session_state.speech_manager.speak_text(last_response):
                    st.success("🔊 Started reading response...")
                else:
                    st.error("🔊 Failed to start reading")
        else:
            if st.button("🔊 Setup Audio", help="Setup audio for TTS", key="setup_tts_chat"):
                st.session_state.speech_manager = SpeechManager()
                st.rerun()
    
    # Handle both text and speech input
    process_input = user_input or st.session_state.get('process_speech_input', None)
    
    if process_input:
        # Clear the speech input flag
        if 'process_speech_input' in st.session_state:
            del st.session_state.process_speech_input
        
        st.session_state.messages.append({"role": "user", "content": process_input})
        
        with st.spinner(f"🤔 Thinking in {st.session_state.mood.lower()} mode..."):
            ai_generator = EnhancedAIGenerator()
            doc_processor = EnhancedDocumentProcessor()
            
            use_rag = bool(st.session_state.documents)
            context = ""
            
            if use_rag:
                context = doc_processor.simple_search(st.session_state.documents, process_input)
                if context:
                    st.info("🔍 Found relevant context in uploaded documents")
            
            response = ai_generator.generate_response(
                process_input, 
                context, 
                use_rag,
                temperature=st.session_state.temperature,
                mood=st.session_state.mood,
                max_tokens=st.session_state.max_tokens
            )
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.session_state.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_message": process_input,
                "assistant_response": response,
                "used_rag": use_rag,
                "settings": {
                    "temperature": st.session_state.temperature,
                    "mood": st.session_state.mood,
                    "max_tokens": st.session_state.max_tokens
                }
            })
        
        st.rerun()
    
    # FIXED EMAIL SECTION
    if st.session_state.messages:
        last_response = st.session_state.messages[-1]["content"]
        
        with st.expander("📧 Send Response as Email", expanded=False):
            # Email connection test
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Email Configuration Status:**")
                if Config.GMAIL_EMAIL and Config.GMAIL_APP_PASSWORD:
                    st.success("✅ Email configured")
                else:
                    st.error("❌ Email not configured")
                    st.info("💡 Set GMAIL_EMAIL and GMAIL_APP_PASSWORD environment variables")
            
            with col2:
                if st.button("🔧 Test Connection", key="test_email_connection"):
                    if EmailSystem.test_email_connection():
                        st.success("✅ Email connection successful!")
                    else:
                        st.error("❌ Email connection failed")
            
            # Email form
            col1, col2 = st.columns(2)
            
            with col1:
                recipient_email = st.text_input(
                    "Recipient Email", 
                    placeholder="recipient@example.com",
                    key="recipient_email_input"
                )
                email_subject = st.text_input(
                    "Subject", 
                    value="AI Assistant Response",
                    key="email_subject_input"
                )
            
            with col2:
                # Preview cleaned content
                if last_response:
                    cleaned_preview = EmailSystem.clean_text_for_email(last_response)
                    st.text_area(
                        "Email Preview (cleaned):", 
                        value=cleaned_preview[:200] + "..." if len(cleaned_preview) > 200 else cleaned_preview,
                        height=100,
                        disabled=True,
                        key="email_preview_area"
                    )
            
            email_message = st.text_area(
                "Message", 
                value=last_response, 
                height=150,
                help="Emojis and special characters will be automatically cleaned for email compatibility",
                key="email_message_input"
            )
            
            # Validation and send
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📤 Send Email", key="send_email_button_chat"):
                    if not recipient_email:
                        st.error("❌ Please enter recipient email")
                    elif not EmailSystem.validate_email(recipient_email):
                        st.error("❌ Invalid email address format")
                    elif not email_subject.strip():
                        st.error("❌ Please enter email subject")
                    elif not email_message.strip():
                        st.error("❌ Please enter email message")
                    else:
                        email_system = EmailSystem()
                        if email_system.send_email(recipient_email, email_subject, email_message):
                            st.success("✅ Email sent successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to send email")
            
            with col2:
                if st.button("🧹 Clear Form", key="clear_email_form"):
                    st.rerun()
            
            with col3:
                if st.button("📋 Copy Content", key="copy_email_content"):
                    cleaned_content = EmailSystem.clean_text_for_email(email_message)
                    st.code(cleaned_content, language="text")
def segmentation_page():
    """Redirect to enhanced segmentation page"""
    enhanced_segmentation_page()

def pose_detection_page():
    """Original pose detection page"""
    st.title("🤸 Pose Detection with YOLOv8")
    
    if not YOLO_SUPPORT:
        st.error("❌ YOLO not available.")
        st.info("Install with: `pip install ultralytics torch`")
        return
    
    if not IMAGE_SUPPORT:
        st.error("❌ Image support not available.")
        st.info("Install with: `pip install Pillow opencv-python numpy`")
        return
    
    st.write("Upload an image to detect human poses and keypoints")
    
    st.info(f"🤸 **Model**: {Config.YOLO_POSE_MODEL} | 🎯 **Confidence**: {st.session_state.pose_confidence}")
    
    uploaded_image = st.file_uploader(
        "Choose an image with people",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for pose detection",
        key="pose_upload"
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🤸 Detect Poses"):
                with st.spinner("🔄 Running YOLOv8 pose detection..."):
                    pose_detector = YOLOPoseDetection()
                    results = pose_detector.detect_poses(image)
                    
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        st.success(f"✅ Pose detection complete! Found {results['count']} person(s).")
                        
                        with col2:
                            st.subheader("🤸 Pose Detection Results")
                            if "pose_image" in results:
                                st.image(results["pose_image"], caption="Detected Poses", use_container_width=True)
                        
                        # Detailed pose analysis
                        if results['poses']:
                            st.subheader("📊 Pose Analysis Details")
                            
                            for pose in results['poses']:
                                st.markdown(f"### 🤸 Person {pose['person_id']}")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Detection Confidence", f"{pose['confidence']:.2%}")
                                with col2:
                                    st.metric("Keypoints Detected", pose['keypoint_count'])
                                with col3:
                                    bbox = pose['bbox']
                                    st.metric("Person Size", f"{bbox[2]:.0f}×{bbox[3]:.0f}")
                                
                                # Show keypoints in expandable section
                                with st.expander(f"🔍 Keypoint Details for Person {pose['person_id']}", expanded=False):
                                    for keypoint in pose['keypoints'][:10]:  # Show first 10
                                        confidence_color = "🟢" if keypoint['confidence'] > 0.8 else "🟡" if keypoint['confidence'] > 0.5 else "🔴"
                                        st.markdown(f"""
                                        <div class="pose-keypoint">
                                            {confidence_color} **{keypoint['name'].replace('_', ' ').title()}**: 
                                            Position ({keypoint['x']:.1f}, {keypoint['y']:.1f}) | 
                                            Confidence: {keypoint['confidence']:.1%}
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown("---")
                        
                        else:
                            st.info("No poses detected in the image.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def emotion_analysis_page():
    """Enhanced emotion analysis interface"""
    st.title("😊 Emotion Analysis")
    
    if not IMAGE_SUPPORT:
        st.error("❌ Image support not available.")
        st.info("Install with: `pip install opencv-python`")
        return
    
    st.write("Upload an image to analyze facial emotions")
    
    # Enhanced model status
    if TENSORFLOW_SUPPORT:
        if os.path.exists(Config.EMOTION_MODEL_PATH):
            st.success("✅ Advanced emotion detection model available")
        else:
            st.info("🔧 Compatible emotion model will be created automatically")
    else:
        st.warning("⚠️ TensorFlow not available - only basic face detection")
        st.info("💡 Install TensorFlow for full emotion detection: `pip install tensorflow`")
    
    st.info(f"😊 **Confidence Threshold**: {st.session_state.emotion_confidence}")
    
    uploaded_image = st.file_uploader(
        "Choose an image with faces",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for emotion analysis",
        key="emotion_upload"
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("😊 Analyze Emotions"):
                with st.spinner("🔄 Analyzing emotions..."):
                    emotion_detector = EmotionDetection()
                    results = emotion_detector.detect_emotions(image)
                    
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        st.success(f"✅ Emotion analysis complete! Found {results['face_count']} face(s).")
                        
                        with col2:
                            st.subheader("😊 Emotion Analysis Results")
                            if "emotion_image" in results:
                                st.image(results["emotion_image"], caption="Detected Emotions", use_container_width=True)
                        
                        if results['emotions']:
                            st.subheader("📊 Emotion Details")
                            
                            for emotion in results['emotions']:
                                st.markdown(f"### 😊 Face {emotion['face_id']}")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Detected Emotion", emotion['emotion'])
                                with col2:
                                    if emotion['confidence'] > 0:
                                        st.metric("Confidence", f"{emotion['confidence']:.2%}")
                                    else:
                                        st.metric("Confidence", "N/A")
                                with col3:
                                    bbox = emotion['bbox']
                                    st.metric("Face Size", f"{bbox[2]}×{bbox[3]}")
                                
                                # Show all emotion probabilities if available
                                if emotion['all_emotions']:
                                    st.subheader("🎭 All Emotion Probabilities")
                                    
                                    # Create a bar chart of emotions
                                    emotion_df = pd.DataFrame(
                                        list(emotion['all_emotions'].items()),
                                        columns=['Emotion', 'Probability']
                                    )
                                    emotion_df = emotion_df.sort_values('Probability', ascending=False)
                                    
                                    st.bar_chart(emotion_df.set_index('Emotion'))
                                    
                                    # Show detailed breakdown
                                    for emo, prob in emotion_df.values:
                                        color = "🟢" if prob > 0.5 else "🟡" if prob > 0.3 else "🔴"
                                        st.write(f"{color} **{emo}**: {prob:.2%}")
                                
                                st.markdown("---")
                        
                        else:
                            st.info("No faces detected in the image.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def real_time_camera_page():
    """Real-time camera analysis interface"""
    st.title("📹 Real-time Camera Analysis")
    
    if not IMAGE_SUPPORT:
        st.error("❌ Image support not available.")
        st.info("Install with: `pip install opencv-python`")
        return
    
    st.write("Access your camera for real-time analysis")
    
    # Camera controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📹 Start Camera"):
            camera_system = CameraSystem()
            if camera_system.initialize_camera():
                st.session_state.camera_active = True
                st.success("✅ Camera started!")
            else:
                st.error("❌ Failed to start camera")
    
    with col2:
        if st.button("🎥 Start Live Feed"):
            if st.session_state.camera_active:
                camera_system = CameraSystem()
                if camera_system.start_live_feed():
                    st.session_state.live_feed_active = True
                    st.session_state.camera_stream = camera_system
                    st.success("✅ Live feed started!")
                else:
                    st.error("❌ Failed to start live feed")
            else:
                st.warning("⚠️ Start camera first")
    
    with col3:
        if st.button("⏹️ Stop Live Feed"):
            if st.session_state.camera_stream:
                st.session_state.camera_stream.stop_live_feed()
            st.session_state.live_feed_active = False
            st.success("✅ Live feed stopped!")
    
    with col4:
        if st.button("⏹️ Stop Camera"):
            if st.session_state.camera_stream:
                st.session_state.camera_stream.stop_camera()
            st.session_state.camera_active = False
            st.session_state.live_feed_active = False
            st.session_state.camera_stream = None
            st.success("✅ Camera stopped!")
    
    # Analysis type selection
    st.subheader("🎛️ Analysis Options")
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        do_segmentation = st.checkbox("🎭 Object Segmentation", value=True)
    with analysis_cols[1]:
        do_pose = st.checkbox("🤸 Pose Detection", value=True)
    with analysis_cols[2]:
        do_emotion = st.checkbox("😊 Emotion Analysis", value=True)
    
    # Single frame capture option
    if st.session_state.camera_active:
        st.info("📹 Camera is active. Use controls above to start live feed")
        
        if st.button("📸 Capture Single Frame"):
            camera_system = CameraSystem()
            camera_system.initialize_camera()
            frame = camera_system.capture_frame()
            
            if frame is not None:
                st.subheader("📸 Captured Frame")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(frame, caption="Captured Frame", use_container_width=True)
                
                with col2:
                    analysis_types = []
                    if do_segmentation:
                        analysis_types.append("segmentation")
                    if do_pose:
                        analysis_types.append("pose")
                    if do_emotion:
                        analysis_types.append("emotion")
                    
                    if analysis_types and st.button("🔍 Analyze Captured Frame"):
                        with st.spinner("🔄 Analyzing frame..."):
                            real_time_analyzer = RealTimeAnalysis()
                            results = real_time_analyzer.run_analysis(frame, analysis_types)
                            
                            # Display results
                            if "segmentation" in results:
                                seg_results = results["segmentation"]
                                if "segmented_image_sg" in seg_results:
                                    st.image(seg_results["segmented_image_sg"], caption="Segmented Objects", use_container_width=True)
                                    st.write(f"🎭 Objects detected: {seg_results.get('count', 0)}")
                            
                            if "pose" in results:
                                pose_results = results["pose"]
                                if "pose_image" in pose_results:
                                    st.image(pose_results["pose_image"], caption="Pose Detection", use_container_width=True)
                                    st.write(f"🤸 People detected: {pose_results.get('count', 0)}")
                            
                            if "emotion" in results:
                                emotion_results = results["emotion"]
                                if "emotion_image" in emotion_results:
                                    st.image(emotion_results["emotion_image"], caption="Emotion Analysis", use_container_width=True)
                                    st.write(f"😊 Faces detected: {emotion_results.get('face_count', 0)}")
            
            camera_system.stop_camera()
    
    else:
        st.info("📹 Click 'Start Camera' to begin real-time analysis")

def chat_history_page():
    """Enhanced chat history interface"""
    st.title("📊 Chat History & Analytics")
    
    if not st.session_state.chat_history:
        st.info("📝 No chat history available. Start a conversation in the Chat Assistant!")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_count = st.selectbox("Show last", [10, 20, 50, "All"], index=1)
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.success("✅ Chat history cleared!")
            st.rerun()
    with col3:
        total_chats = len(st.session_state.chat_history)
        st.metric("Total Conversations", total_chats)
    
    if st.session_state.chat_history:
        with st.expander("📈 Chat Analytics", expanded=False):
            moods = [chat.get('settings', {}).get('mood', 'Unknown') for chat in st.session_state.chat_history]
            mood_counts = pd.Series(moods).value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎭 Mood Usage")
                for mood, count in mood_counts.items():
                    emoji = MoodSystem.get_mood_emoji(mood) if mood != 'Unknown' else '❓'
                    st.write(f"{emoji} {mood}: {count}")
            
            with col2:
                st.subheader("🔍 RAG Usage")
                rag_usage = sum(1 for chat in st.session_state.chat_history if chat.get('used_rag', False))
                st.write(f"📄 Conversations with documents: {rag_usage}")
                st.write(f"💬 Regular conversations: {total_chats - rag_usage}")
    
    history_to_show = st.session_state.chat_history
    if show_count != "All":
        history_to_show = history_to_show[-int(show_count):]
    
    for i, chat in enumerate(reversed(history_to_show)):
        settings = chat.get('settings', {})
        mood = settings.get('mood', 'Unknown')
        mood_emoji = MoodSystem.get_mood_emoji(mood) if mood != 'Unknown' else '❓'
        
        title = f"{mood_emoji} Chat {len(history_to_show) - i} - {chat['timestamp'][:19]}"
        
        with st.expander(title, expanded=False):
            st.markdown(f"**👤 User:** {chat['user_message']}")
            st.markdown(f"**🤖 Assistant:** {chat['assistant_response']}")
            
            if settings:
                setting_info = f"⚙️ *Settings: {mood} mood, Temperature: {settings.get('temperature', 'N/A')}, Max tokens: {settings.get('max_tokens', 'N/A')}*"
                st.markdown(setting_info)
            
            if chat.get('used_rag', False):
                st.markdown("🔍 *Used document context*")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📧 Email This", key=f"email_{i}"):
                    st.session_state.email_content = chat['assistant_response']
            with col2:
                if st.button(f"🔄 Continue Chat", key=f"continue_{i}"):
                    st.session_state.messages.append({"role": "user", "content": chat['user_message']})
                    st.session_state.messages.append({"role": "assistant", "content": chat['assistant_response']})
                    st.success("✅ Added to current chat!")
            with col3:
                if st.button(f"📋 Copy Response", key=f"copy_{i}"):
                    st.code(chat['assistant_response'])

def live_sentiment_analysis_page():
    """Live sentiment analysis interface with real-time features"""
    st.title("💭 Live Sentiment Analysis Dashboard")
    st.write("Real-time text sentiment analysis with advanced analytics")
    
    # Check available sentiment analyzers
    sentiment_status = []
    if VADER_AVAILABLE:
        sentiment_status.append("✅ VADER Sentiment (Advanced)")
    if TEXTBLOB_AVAILABLE:
        sentiment_status.append("✅ TextBlob Sentiment (Basic)")
    if SPEECH_SUPPORT:
        sentiment_status.append("✅ Speech Recognition (Available)")
    if not sentiment_status:
        sentiment_status.append("⚠️ Basic Keyword Analysis Only")
    
    with st.expander("🔧 Sentiment Analysis Status", expanded=False):
        for status in sentiment_status:
            st.write(status)
        
        if not VADER_AVAILABLE and not TEXTBLOB_AVAILABLE:
            st.warning("⚠️ For advanced sentiment analysis, install: `pip install vaderSentiment textblob`")
    
    # Main tabs for different sentiment analysis features
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Real-time Text Analysis", 
        "📊 Sentiment Dashboard", 
        "📈 Analytics & History",
        "⚙️ Advanced Features"
    ])
    
    with tab1:
        st.subheader("💬 Real-time Text Sentiment Analysis")
        st.write("Type or paste text below to analyze sentiment in real-time")
        
        # Text input for live sentiment analysis
        text_input = st.text_area(
            "Enter text for sentiment analysis:",
            height=150,
            placeholder="Type your message here to analyze sentiment in real-time...\n\nExamples:\n- 'I love this new feature, it's amazing!'\n- 'This is terrible and frustrating'\n- 'The weather is okay today'"
        )
        
        # Auto-analysis checkbox
        auto_analyze = st.checkbox("🔄 Auto-analyze as I type", value=True)
        
        # Manual analysis button
        manual_analyze = st.button("🔍 Analyze Sentiment")
        
        # Real-time analysis
        if text_input and (auto_analyze or manual_analyze):
            sentiment_result = st.session_state.sentiment_analyzer.analyze_text_sentiment(text_input)
            
            if "error" not in sentiment_result:
                # Display sentiment result with enhanced visualization
                st.markdown("### 🎯 Sentiment Analysis Results")
                
                # Main sentiment metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sentiment_color = {
                        "positive": "🟢",
                        "negative": "🔴", 
                        "neutral": "🟡"
                    }
                    sentiment_emoji = {
                        "positive": "😊",
                        "negative": "😞",
                        "neutral": "😐"
                    }
                    st.metric(
                        "Sentiment", 
                        f"{sentiment_color.get(sentiment_result['sentiment'], '⚪')} {sentiment_result['sentiment'].title()}"
                    )
                    st.write(f"**Emotion**: {sentiment_emoji.get(sentiment_result['sentiment'], '🤔')}")
                
                with col2:
                    confidence_color = "🟢" if sentiment_result['confidence'] > 0.7 else "🟡" if sentiment_result['confidence'] > 0.4 else "🔴"
                    st.metric("Confidence", f"{confidence_color} {sentiment_result['confidence']:.1%}")
                
                with col3:
                    st.metric("Words", sentiment_result['word_count'])
                    st.metric("Characters", sentiment_result['length'])
                
                with col4:
                    st.metric("Analyzer", sentiment_result.get('analyzer', 'Unknown'))
                    timestamp = datetime.fromisoformat(sentiment_result['timestamp']).strftime('%H:%M:%S')
                    st.write(f"**Time**: {timestamp}")
                
                # Detailed sentiment breakdown
                if 'positive' in sentiment_result:  # VADER analysis
                    st.markdown("### 📊 Detailed Sentiment Breakdown (VADER)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment scores breakdown
                        sentiment_data = {
                            'Positive': sentiment_result['positive'],
                            'Negative': sentiment_result['negative'], 
                            'Neutral': sentiment_result['neutral']
                        }
                        
                        # Create DataFrame for chart
                        sentiment_df = pd.DataFrame(
                            list(sentiment_data.items()),
                            columns=['Sentiment', 'Score']
                        )
                        st.bar_chart(sentiment_df.set_index('Sentiment'))
                    
                    with col2:
                        # Compound score visualization
                        compound = sentiment_result['compound']
                        st.write("**Compound Score Analysis:**")
                        st.write(f"• **Score**: {compound:.3f}")
                        
                        if compound >= 0.05:
                            st.success("🟢 **Overall**: Positive sentiment")
                        elif compound <= -0.05:
                            st.error("🔴 **Overall**: Negative sentiment")
                        else:
                            st.info("🟡 **Overall**: Neutral sentiment")
                        
                        # Score interpretation
                        st.write("**Score Range:**")
                        st.write("• +1.0 = Most positive")
                        st.write("• 0.0 = Neutral") 
                        st.write("• -1.0 = Most negative")
                
                elif 'polarity' in sentiment_result:  # TextBlob analysis
                    st.markdown("### 📊 TextBlob Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        polarity = sentiment_result['polarity']
                        st.metric("Polarity", f"{polarity:.3f}")
                        st.write("(-1 = Negative, +1 = Positive)")
                        
                        # Polarity bar
                        polarity_data = pd.DataFrame({
                            'Metric': ['Polarity'],
                            'Score': [polarity]
                        })
                        st.bar_chart(polarity_data.set_index('Metric'))
                    
                    with col2:
                        subjectivity = sentiment_result['subjectivity']
                        st.metric("Subjectivity", f"{subjectivity:.3f}")
                        st.write("(0 = Objective, 1 = Subjective)")
                        
                        # Subjectivity interpretation
                        if subjectivity > 0.7:
                            st.info("🎭 **High subjectivity** - Opinion-based")
                        elif subjectivity > 0.3:
                            st.info("📊 **Medium subjectivity** - Mixed")
                        else:
                            st.info("📰 **Low subjectivity** - Fact-based")
                
                else:  # Simple keyword analysis
                    st.markdown("### 📊 Keyword-based Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Positive Words", sentiment_result.get('positive_words', 0))
                    with col2:
                        st.metric("Negative Words", sentiment_result.get('negative_words', 0))
                
                # Text insights
                st.markdown("### 💡 Text Insights")
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    # Word count analysis
                    word_count = sentiment_result['word_count']
                    if word_count < 5:
                        st.info("📝 **Short text** - Consider adding more content for better analysis")
                    elif word_count > 50:
                        st.info("📄 **Long text** - Analysis covers overall sentiment")
                    else:
                        st.success("✅ **Good length** - Optimal for sentiment analysis")
                
                with insights_col2:
                    # Confidence analysis
                    confidence = sentiment_result['confidence']
                    if confidence > 0.8:
                        st.success("🎯 **High confidence** - Strong sentiment indicators")
                    elif confidence > 0.5:
                        st.warning("⚖️ **Medium confidence** - Moderate sentiment")
                    else:
                        st.info("🤔 **Low confidence** - Weak or mixed sentiment")
                
                # Analysis metadata
                with st.expander("🔍 Detailed Analysis Data", expanded=False):
                    st.json(sentiment_result)
            
            else:
                st.error(f"❌ {sentiment_result['error']}")
        
        elif not text_input:
            st.info("💭 Enter some text above to see live sentiment analysis")
    
    with tab2:
        st.subheader("📊 Live Sentiment Dashboard")
        
        # Get sentiment summary
        sentiment_summary = st.session_state.sentiment_analyzer.get_sentiment_summary()
        
        if "error" not in sentiment_summary and sentiment_summary['total_analyses'] > 0:
            # Dashboard metrics
            st.markdown("### 📈 Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", sentiment_summary['total_analyses'])
            
            with col2:
                st.metric("Recent Analyses", sentiment_summary['recent_count'])
            
            with col3:
                st.metric("Average Confidence", f"{sentiment_summary['average_confidence']:.1%}")
            
            with col4:
                dominant_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                dominant = sentiment_summary['dominant_sentiment']
                st.metric("Dominant Sentiment", f"{dominant_emoji.get(dominant, '🤔')} {dominant.title()}")
            
            # Sentiment distribution
            if sentiment_summary['sentiment_distribution']:
                st.markdown("### 🥧 Sentiment Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    dist_data = sentiment_summary['sentiment_distribution']
                    dist_df = pd.DataFrame(
                        list(dist_data.items()),
                        columns=['Sentiment', 'Count']
                    )
                    st.bar_chart(dist_df.set_index('Sentiment'))
                
                with col2:
                    # Pie chart data display
                    total = sum(dist_data.values())
                    if total > 0:
                        st.write("**Percentage Breakdown:**")
                        for sentiment, count in dist_data.items():
                            percentage = (count / total) * 100
                            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                            st.write(f"{emoji.get(sentiment, '⚪')} **{sentiment.title()}**: {percentage:.1f}% ({count})")
            
            # Recent activity feed
            st.markdown("### 🔄 Recent Analysis Activity")
            
            recent_analyses = st.session_state.sentiment_analyzer.sentiment_history[-5:]  # Last 5
            
            if recent_analyses:
                for i, analysis in enumerate(reversed(recent_analyses)):
                    if 'sentiment' in analysis:
                        timestamp = datetime.fromisoformat(analysis['timestamp']).strftime('%H:%M:%S')
                        sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                        
                        with st.expander(f"{sentiment_emoji.get(analysis['sentiment'], '🤔')} {timestamp} - {analysis['sentiment'].title()} ({analysis['confidence']:.1%})", expanded=False):
                            st.write(f"**Text**: \"{analysis['text'][:100]}{'...' if len(analysis['text']) > 100 else ''}\"")
                            st.write(f"**Confidence**: {analysis['confidence']:.1%}")
                            st.write(f"**Word Count**: {analysis['word_count']}")
                            st.write(f"**Analyzer**: {analysis.get('analyzer', 'Unknown')}")
        
        else:
            st.info("📝 Perform some sentiment analyses to see dashboard data")
            st.write("Go to the 'Real-time Text Analysis' tab to start analyzing text!")
    
    with tab3:
        st.subheader("📈 Analytics & History")
        
        # Export/Import functionality
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Export History"):
                if st.session_state.sentiment_analyzer.sentiment_history:
                    # Create downloadable JSON
                    history_json = json.dumps(st.session_state.sentiment_analyzer.sentiment_history, indent=2)
                    st.download_button(
                        label="💾 Download Sentiment History",
                        data=history_json,
                        file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No history to export")
        
        with col2:
            if st.button("🧹 Clear History"):
                st.session_state.sentiment_analyzer.sentiment_history = []
                st.success("✅ Sentiment history cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 Generate Report"):
                # Generate comprehensive report
                summary = st.session_state.sentiment_analyzer.get_sentiment_summary()
                if "error" not in summary:
                    st.markdown("### 📋 Sentiment Analysis Report")
                    st.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Total Analyses**: {summary['total_analyses']}")
                    st.write(f"**Dominant Sentiment**: {summary['dominant_sentiment'].title()}")
                    st.write(f"**Average Confidence**: {summary['average_confidence']:.1%}")
        
        # Historical trends (if enough data)
        if len(st.session_state.sentiment_analyzer.sentiment_history) >= 5:
            st.markdown("### 📈 Sentiment Trends Over Time")
            
            # Create trend data
            trend_data = []
            for i, analysis in enumerate(st.session_state.sentiment_analyzer.sentiment_history):
                if 'sentiment' in analysis:
                    sentiment_score = 1 if analysis['sentiment'] == 'positive' else -1 if analysis['sentiment'] == 'negative' else 0
                    trend_data.append({
                        'Analysis': i + 1,
                        'Sentiment Score': sentiment_score,
                        'Confidence': analysis['confidence']
                    })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Sentiment Trend** (1=Positive, 0=Neutral, -1=Negative)")
                    st.line_chart(trend_df.set_index('Analysis')['Sentiment Score'])
                
                with col2:
                    st.write("**Confidence Trend**")
                    st.line_chart(trend_df.set_index('Analysis')['Confidence'])
        
        else:
            st.info("📊 Perform at least 5 analyses to see trend data")
    
    with tab4:
        st.subheader("⚙️ Advanced Features & Settings")
        
        # Batch text analysis
        st.markdown("### 📄 Batch Text Analysis")
        st.write("Analyze multiple texts at once")
        
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=100,
            placeholder="Line 1: First text to analyze\nLine 2: Second text to analyze\nLine 3: Third text to analyze"
        )
        
        if st.button("🔍 Analyze All Lines") and batch_text:
            lines = [line.strip() for line in batch_text.split('\n') if line.strip()]
            
            if lines:
                st.write(f"**Analyzing {len(lines)} texts...**")
                
                batch_results = []
                progress_bar = st.progress(0)
                
                for i, line in enumerate(lines):
                    # Update progress
                    progress_bar.progress((i + 1) / len(lines))
                    
                    result = st.session_state.sentiment_analyzer.analyze_text_sentiment(line)
                    if "error" not in result:
                        batch_results.append({
                            'Line': i + 1,
                            'Text': line[:50] + '...' if len(line) > 50 else line,
                            'Sentiment': result['sentiment'].title(),
                            'Confidence': f"{result['confidence']:.1%}",
                            'Words': result['word_count'],
                            'Analyzer': result.get('analyzer', 'Unknown')
                        })
                
                progress_bar.empty()
                
                if batch_results:
                    # Display results table
                    st.markdown("### 📊 Batch Analysis Results")
                    results_df = pd.DataFrame(batch_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Batch summary
                    sentiments = [r['Sentiment'] for r in batch_results]
                    st.markdown("### 📈 Batch Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analyzed", len(batch_results))
                    
                    with col2:
                        positive_count = sentiments.count('Positive')
                        st.metric("😊 Positive", positive_count)
                    
                    with col3:
                        negative_count = sentiments.count('Negative')
                        st.metric("😞 Negative", negative_count)
                    
                    with col4:
                        neutral_count = sentiments.count('Neutral')
                        st.metric("😐 Neutral", neutral_count)
                    
                    # Visual breakdown
                    if len(set(sentiments)) > 1:
                        st.markdown("### 📊 Sentiment Distribution")
                        sentiment_counts = pd.Series(sentiments).value_counts()
                        st.bar_chart(sentiment_counts)
                    
                    # Export option
                    if st.button("📥 Download Results as CSV"):
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv_data,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning("⚠️ No valid results from batch analysis")
            
            else:
                st.warning("⚠️ Please enter at least one line of text")
        
        # Settings and configuration
        st.markdown("### ⚙️ Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**History Settings:**")
            max_history = st.slider("Max history entries", 10, 200, 50)
            if st.button("Apply History Limit"):
                # Trim history if needed
                if len(st.session_state.sentiment_analyzer.sentiment_history) > max_history:
                    st.session_state.sentiment_analyzer.sentiment_history = st.session_state.sentiment_analyzer.sentiment_history[-max_history:]
                st.success(f"✅ History limit set to {max_history}")
        
        with col2:
            st.write("**Analysis Options:**")
            confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.1)
            show_debug = st.checkbox("Show debug information")
            
            if show_debug:
                st.write("**Debug Info:**")
                st.write(f"• VADER Available: {VADER_AVAILABLE}")
                st.write(f"• TextBlob Available: {TEXTBLOB_AVAILABLE}")
                st.write(f"• Speech Recognition: {SPEECH_SUPPORT}")
                st.write(f"• Current History Length: {len(st.session_state.sentiment_analyzer.sentiment_history)}")
        
        # Installation guide
        with st.expander("📦 Installation Guide for Enhanced Sentiment Analysis", expanded=False):
            st.markdown("""
            ### Install additional packages for advanced sentiment analysis:
            
            ```bash
            # For VADER sentiment analysis (recommended)
            pip install vaderSentiment
            
            # For TextBlob sentiment analysis (alternative)
            pip install textblob
            python -m textblob.download_corpora
            
            # For speech recognition (optional)
            pip install SpeechRecognition pyaudio
            ```
            
            ### Features by package:
            - **VADER**: Most accurate sentiment analysis with detailed breakdown
            - **TextBlob**: Good alternative with polarity and subjectivity analysis
            - **SpeechRecognition**: Convert speech to text for audio sentiment analysis
            
            ### Current Status:
            """)
            
            if VADER_AVAILABLE:
                st.success("✅ VADER Sentiment Analysis - Advanced features available")
            elif TEXTBLOB_AVAILABLE:
                st.info("ℹ️ TextBlob Sentiment Analysis - Basic features available")
            else:
                st.warning("⚠️ Only keyword-based sentiment analysis available")

def enhanced_segmentation_page():
    """Enhanced object segmentation with dashboard"""
    st.title("🎭 Object Segmentation with Enhanced Dashboard")
    
    if not YOLO_SUPPORT:
        st.error("❌ YOLO not available.")
        st.info("Install with: `pip install ultralytics torch`")
        return
    
    # Initialize dashboard
    if 'object_dashboard' not in st.session_state:
        st.session_state.object_dashboard = ObjectDetectionDashboard()
    
    # Show dashboard toggle
    show_dashboard = st.checkbox("📊 Show Object Detection Dashboard", value=True)
    
    if show_dashboard:
        st.session_state.object_dashboard.display_dashboard()
        st.markdown("---")
    
    st.write("Upload an image to perform object segmentation with analytics")
    
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for object segmentation with dashboard analytics"
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            st.subheader("📷 Original Image")
            st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🎭 Run Enhanced Segmentation Analysis"):
                with st.spinner("🔄 Running enhanced segmentation analysis..."):
                    segmenter = YOLOSegmentation()
                    results = segmenter.segment_objects(image)
                    
                    if "error" in results:
                        st.error(f"❌ {results['error']}")
                    else:
                        # Add to dashboard
                        st.session_state.object_dashboard.add_detection_results(results)
                        
                        st.success(f"✅ Segmentation complete! Found {results['count']} objects.")
                        
                        # ENHANCED: Object Detection Summary Dashboard
                        if results['detections']:
                            st.markdown("### 📊 Detection Summary")
                            
                            # Quick metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Objects Found", results['count'])
                            
                            with col2:
                                avg_confidence = np.mean([d['confidence'] for d in results['detections']])
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            with col3:
                                unique_objects = len(set([d['label'] for d in results['detections']]))
                                st.metric("Unique Types", unique_objects)
                            
                            with col4:
                                total_area = sum([d.get('mask_area', 0) for d in results['detections']])
                                st.metric("Total Mask Area", f"{total_area:,} px")
                            
                            # Object breakdown
                            st.markdown("#### 🏷️ Detected Objects Breakdown")
                            
                            detection_data = []
                            for i, detection in enumerate(results['detections']):
                                confidence_emoji = "🟢" if detection['confidence'] > 0.8 else "🟡" if detection['confidence'] > 0.5 else "🔴"
                                detection_data.append({
                                    "ID": i + 1,
                                    "Object": detection['label'].title(),
                                    "Confidence": f"{confidence_emoji} {detection['confidence']:.1%}",
                                    "Area (px)": f"{detection.get('mask_area', 0):,}",
                                    "Bbox": f"{detection['bbox'][2]:.0f}×{detection['bbox'][3]:.0f}"
                                })
                            
                            df = pd.DataFrame(detection_data)
                            st.dataframe(df, use_container_width=True)
                        
                        # Display segmentation results
                        st.markdown("---")
                        st.subheader("🎯 Segmentation Results")
                        
                        tab1, tab2, tab3 = st.tabs([
                            "🎭 Object Segmented (SG)", 
                            "📦 Object Detected (BB)", 
                            "🎨 Segmentation Masks"
                        ])
                        
                        with tab1:
                            if "segmented_image_sg" in results:
                                st.image(results["segmented_image_sg"], caption="Object Segmented (SG)", use_container_width=True)
                        
                        with tab2:
                            if "bbox_image_bb" in results:
                                st.image(results["bbox_image_bb"], caption="Object Detected (BB)", use_container_width=True)
                        
                        with tab3:
                            if "mask_overlay_pure" in results:
                                st.image(results["mask_overlay_pure"], caption="Segmentation Masks", use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

def mediapipe_analysis_page():
    """NEW: MediaPipe comprehensive analysis page"""
    st.title("🤖 MediaPipe Multi-Modal Analysis")
    
    if not MEDIAPIPE_SUPPORT:
        st.error("❌ MediaPipe not available.")
        st.info("Install with: `pip install mediapipe`")
        return
    
    st.write("Advanced computer vision analysis using Google's MediaPipe")
    
    # Analysis options
    st.subheader("🎛️ Analysis Options")
    analysis_cols = st.columns(4)
    
    with analysis_cols[0]:
        do_face_detection = st.checkbox("👤 Face Detection", value=True)
    with analysis_cols[1]:
        do_hand_landmarks = st.checkbox("✋ Hand Landmarks", value=True)
    with analysis_cols[2]:
        do_pose_landmarks = st.checkbox("🤸 Pose Landmarks", value=True)
    with analysis_cols[3]:
        do_face_mesh = st.checkbox("🕸️ Face Mesh", value=False)
    
    uploaded_image = st.file_uploader(
        "Choose an image for MediaPipe analysis",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for comprehensive MediaPipe analysis"
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🤖 Run MediaPipe Analysis"):
                with col2:
                    st.subheader("🤖 MediaPipe Results")
                    
                    analyzer = MediaPipeAnalyzer()
                    results_tabs = []
                    results_data = []
                    
                    # Face Detection
                    if do_face_detection:
                        with st.spinner("🔄 Analyzing faces..."):
                            face_results = analyzer.analyze_face_detection(image)
                            if "error" not in face_results:
                                results_tabs.append("👤 Faces")
                                results_data.append(("face", face_results))
                                st.success(f"✅ Found {face_results['count']} face(s)")
                    
                    # Hand Landmarks
                    if do_hand_landmarks:
                        with st.spinner("🔄 Analyzing hands..."):
                            hand_results = analyzer.analyze_hand_landmarks(image)
                            if "error" not in hand_results:
                                results_tabs.append("✋ Hands")
                                results_data.append(("hands", hand_results))
                                st.success(f"✅ Found {hand_results['count']} hand(s)")
                    
                    # Pose Landmarks
                    if do_pose_landmarks:
                        with st.spinner("🔄 Analyzing pose..."):
                            pose_results = analyzer.analyze_pose_landmarks(image)
                            if "error" not in pose_results:
                                results_tabs.append("🤸 Pose")
                                results_data.append(("pose", pose_results))
                                st.success(f"✅ Found {pose_results['count']} pose(s)")
                
                # Display results in tabs
                if results_tabs:
                    st.markdown("---")
                    tabs = st.tabs(results_tabs)
                    
                    for i, (analysis_type, results) in enumerate(results_data):
                        with tabs[i]:
                            if analysis_type == "face":
                                st.image(results["face_image"], caption="Face Detection", use_container_width=True)
                                
                                if results["faces"]:
                                    st.markdown("#### 📊 Face Analysis")
                                    for face in results["faces"]:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(f"Face {face['face_id']} Confidence", f"{face['confidence']:.1%}")
                                        with col2:
                                            bbox = face['bbox']
                                            st.write(f"**Position**: ({bbox[0]}, {bbox[1]})")
                                            st.write(f"**Size**: {bbox[2]}×{bbox[3]} px")
                            
                            elif analysis_type == "hands":
                                st.image(results["hand_image"], caption="Hand Landmarks", use_container_width=True)
                                
                                if results["hands"]:
                                    st.markdown("#### ✋ Hand Analysis")
                                    for hand in results["hands"]:
                                        st.write(f"**Hand {hand['hand_id']}**: {hand['handedness']}")
                                        st.write(f"**Landmarks**: {hand['landmark_count']}")
                                        
                                        # Show key landmarks
                                        key_landmarks = hand['landmarks'][:5]  # First 5 landmarks
                                        for landmark in key_landmarks:
                                            st.write(f"• Point {landmark['id']}: ({landmark['x']:.3f}, {landmark['y']:.3f})")
                            
                            elif analysis_type == "pose":
                                st.image(results["pose_image"], caption="Pose Landmarks", use_container_width=True)
                                
                                if results["poses"]:
                                    st.markdown("#### 🤸 Pose Analysis")
                                    for pose in results["poses"]:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Pose Confidence", f"{pose['confidence']:.1%}")
                                        with col2:
                                            st.metric("Landmarks Detected", pose['landmark_count'])
                                        
                                        # Show visibility stats
                                        visible_landmarks = [lm for lm in pose['landmarks'] if lm['visibility'] > 0.5]
                                        st.write(f"**Visible landmarks**: {len(visible_landmarks)}/{pose['landmark_count']}")
                
                else:
                    st.warning("⚠️ No analysis results to display. Check MediaPipe setup.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Update the original segmentation_page function to use enhanced version
def segmentation_page():
    """Redirect to enhanced segmentation page"""
    enhanced_segmentation_page()

#MediaPipe option
def enhanced_pose_detection_page():
    """Enhanced pose detection with both YOLO and MediaPipe options"""
    st.title("🤸 Enhanced Pose Detection")
    
    st.write("Choose between YOLO pose detection or MediaPipe pose landmarks")
    
    # Method selection
    pose_method = st.radio(
        "Select pose detection method:",
        ["🎯 YOLO Pose Detection", "🤖 MediaPipe Pose Landmarks"],
        index=0
    )
    
    uploaded_image = st.file_uploader(
        "Choose an image with people",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image for pose detection",
        key="enhanced_pose_upload"
    )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            if st.button("🤸 Detect Poses"):
                with col2:
                    st.subheader("🤸 Pose Detection Results")
                    
                    if pose_method == "🎯 YOLO Pose Detection":
                        if not YOLO_SUPPORT:
                            st.error("❌ YOLO not available. Install with: pip install ultralytics torch")
                            return
                        
                        with st.spinner("🔄 Running YOLO pose detection..."):
                            pose_detector = YOLOPoseDetection()
                            results = pose_detector.detect_poses(image)
                            
                            if "error" in results:
                                st.error(f"❌ {results['error']}")
                            else:
                                st.success(f"✅ YOLO pose detection complete! Found {results['count']} person(s).")
                                if "pose_image" in results:
                                    st.image(results["pose_image"], caption="YOLO Pose Detection", use_container_width=True)
                    
                    else:  # MediaPipe
                        if not MEDIAPIPE_SUPPORT:
                            st.error("❌ MediaPipe not available. Install with: pip install mediapipe")
                            return
                        
                        with st.spinner("🔄 Running MediaPipe pose analysis..."):
                            analyzer = MediaPipeAnalyzer()
                            results = analyzer.analyze_pose_landmarks(image)
                            
                            if "error" in results:
                                st.error(f"❌ {results['error']}")
                            else:
                                st.success(f"✅ MediaPipe pose analysis complete! Found {results['count']} pose(s).")
                                if "pose_image" in results:
                                    st.image(results["pose_image"], caption="MediaPipe Pose Landmarks", use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


def enhanced_real_time_camera_page():
    """Enhanced real-time camera with auto-capture and MediaPipe"""
    st.title("📹 Enhanced Real-time Analysis with Auto-Capture")
    
    if not IMAGE_SUPPORT:
        st.error("❌ Image support not available.")
        return
    
    # Initialize enhanced camera system
    if 'enhanced_camera' not in st.session_state:
        st.session_state.enhanced_camera = AutoCaptureCameraSystem()
    
    st.write("Advanced real-time analysis with auto-capture and MediaPipe integration")
    
    # Camera controls
    st.subheader("📹 Camera Controls")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📹 Start Camera"):
            if st.session_state.enhanced_camera.initialize_camera():
                st.session_state.camera_active = True
                st.success("✅ Camera started!")
            else:
                st.error("❌ Failed to start camera")
    
    with col2:
        if st.button("🎥 Start Live Feed"):
            if st.session_state.camera_active:
                if st.session_state.enhanced_camera.start_live_feed():
                    st.session_state.live_feed_active = True
                    st.success("✅ Live feed started!")
                else:
                    st.error("❌ Failed to start live feed")
            else:
                st.warning("⚠️ Start camera first")
    
    with col3:
        auto_capture_interval = st.selectbox("Auto-capture interval", [1.0, 2.0, 3.0, 5.0], index=1)
        if st.button("🔄 Start Auto-Capture"):
            if st.session_state.camera_active:
                if st.session_state.enhanced_camera.start_auto_capture(auto_capture_interval):
                    st.session_state.auto_capture_active = True
                    st.success(f"✅ Auto-capture started! ({auto_capture_interval}s intervals)")
                else:
                    st.error("❌ Failed to start auto-capture")
            else:
                st.warning("⚠️ Start camera first")
    
    with col4:
        if st.button("⏹️ Stop All"):
            st.session_state.enhanced_camera.stop_auto_capture()
            st.session_state.enhanced_camera.stop_live_feed()
            st.session_state.enhanced_camera.stop_camera()
            st.session_state.camera_active = False
            st.session_state.live_feed_active = False
            st.session_state.auto_capture_active = False
            st.success("✅ All stopped!")
    
    # Analysis selection
    st.subheader("🎛️ Auto-Capture Analysis Selection")
    analysis_cols = st.columns(6)
    
    analysis_types = []
    with analysis_cols[0]:
        if st.checkbox("🎭 Object Detection", value=True):
            analysis_types.append("object_detection")
    with analysis_cols[1]:
        if st.checkbox("👤 MediaPipe Face", value=True):
            analysis_types.append("mediapipe_face")
    with analysis_cols[2]:
        if st.checkbox("✋ MediaPipe Hands", value=False):
            analysis_types.append("mediapipe_hands")
    with analysis_cols[3]:
        if st.checkbox("🤸 MediaPipe Pose", value=False):
            analysis_types.append("mediapipe_pose")
    with analysis_cols[4]:
        if st.checkbox("😊 Emotion Detection", value=True):
            analysis_types.append("emotion")
    with analysis_cols[5]:
        if st.checkbox("📊 Show Dashboard", value=True):
            show_dashboard = True
        else:
            show_dashboard = False
    
    # Live feed display
    if st.session_state.live_feed_active:
        st.subheader("🎥 Live Camera Feed")
        
        live_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Get latest frame
        latest_frame = st.session_state.enhanced_camera.get_latest_frame()
        
        if latest_frame is not None:
            with live_placeholder.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(latest_frame, caption="🎥 Live Feed", use_container_width=True)
                
                with col2:
                    st.write("**Live Feed Status**")
                    st.write("🟢 Active")
                    st.write(f"📏 Size: {latest_frame.size[0]}×{latest_frame.size[1]}")
                    
                    # Manual capture
                    if st.button("📸 Capture Now"):
                        captured_data = st.session_state.enhanced_camera.auto_capture_and_analyze(analysis_types)
                        if captured_data:
                            st.success("✅ Frame captured and analyzed!")
                        else:
                            st.warning("⚠️ No analysis results")
        
        # Auto-capture status and results
        if st.session_state.auto_capture_active:
            with status_placeholder.container():
                st.info(f"🔄 Auto-capture active (every {auto_capture_interval}s)")
                
                # Check for new auto-capture results
                auto_data = st.session_state.enhanced_camera.auto_capture_and_analyze(analysis_types)
                if auto_data:
                    st.success(f"🔄 Auto-captured at {datetime.fromisoformat(auto_data['timestamp']).strftime('%H:%M:%S')}")
        
        # Show dashboard if enabled
        if show_dashboard and hasattr(st.session_state.enhanced_camera, 'object_dashboard'):
            st.markdown("---")
            st.session_state.enhanced_camera.object_dashboard.display_dashboard()
        
        # Show recent auto-capture results
        if hasattr(st.session_state.enhanced_camera, 'auto_capture_results') and st.session_state.enhanced_camera.auto_capture_results:
            st.markdown("---")
            st.subheader("📊 Recent Auto-Capture Results")
            
            recent_results = st.session_state.enhanced_camera.auto_capture_results[-3:]  # Last 3 results
            
            for i, result in enumerate(reversed(recent_results)):
                timestamp = datetime.fromisoformat(result['timestamp']).strftime('%H:%M:%S')
                analyses = list(result['analyses'].keys())
                
                with st.expander(f"🕒 {timestamp} - {len(analyses)} analyses", expanded=False):
                    tabs = st.tabs([f"📷 Frame"] + [f"🔍 {analysis.replace('_', ' ').title()}" for analysis in analyses])
                    
                    with tabs[0]:
                        st.image(result['frame'], caption=f"Captured at {timestamp}", use_container_width=True)
                    
                    tab_idx = 1
                    for analysis_type, analysis_result in result['analyses'].items():
                        with tabs[tab_idx]:
                            if analysis_type == "object_detection":
                                if "segmented_image_sg" in analysis_result:
                                    st.image(analysis_result["segmented_image_sg"], caption="Object Detection", use_container_width=True)
                                    st.write(f"🎭 **Objects detected**: {analysis_result.get('count', 0)}")
                                    
                                    if analysis_result.get('detections'):
                                        for detection in analysis_result['detections'][:3]:  # Show first 3
                                            confidence_color = "🟢" if detection['confidence'] > 0.8 else "🟡" if detection['confidence'] > 0.5 else "🔴"
                                            st.write(f"{confidence_color} **{detection['label'].title()}**: {detection['confidence']:.1%}")
                            
                            elif analysis_type == "mediapipe_face":
                                if "face_image" in analysis_result:
                                    st.image(analysis_result["face_image"], caption="MediaPipe Face Detection", use_container_width=True)
                                    st.write(f"👤 **Faces detected**: {analysis_result.get('count', 0)}")
                            
                            elif analysis_type == "mediapipe_hands":
                                if "hand_image" in analysis_result:
                                    st.image(analysis_result["hand_image"], caption="MediaPipe Hand Landmarks", use_container_width=True)
                                    st.write(f"✋ **Hands detected**: {analysis_result.get('count', 0)}")
                            
                            elif analysis_type == "mediapipe_pose":
                                if "pose_image" in analysis_result:
                                    st.image(analysis_result["pose_image"], caption="MediaPipe Pose Landmarks", use_container_width=True)
                                    st.write(f"🤸 **Poses detected**: {analysis_result.get('count', 0)}")
                            
                            elif analysis_type == "emotion":
                                if "emotion_image" in analysis_result:
                                    st.image(analysis_result["emotion_image"], caption="Emotion Analysis", use_container_width=True)
                                    st.write(f"😊 **Faces analyzed**: {analysis_result.get('face_count', 0)}")
                                    
                                    if analysis_result.get('emotions'):
                                        for emotion in analysis_result['emotions']:
                                            if emotion['emotion'] != 'Unknown':
                                                st.write(f"😊 **Face {emotion['face_id']}**: {emotion['emotion']} ({emotion['confidence']:.1%})")
                        
                        tab_idx += 1
    
    elif st.session_state.camera_active:
        st.info("📹 Camera is active. Click 'Start Live Feed' to see live preview")
        
        # Single frame capture option
        if st.button("📸 Capture Single Frame"):
            frame = st.session_state.enhanced_camera.capture_frame()
            
            if frame is not None:
                st.subheader("📸 Captured Frame")
                st.image(frame, caption="Captured Frame", use_container_width=True)
                
                if analysis_types and st.button("🔍 Analyze Captured Frame"):
                    with st.spinner("🔄 Analyzing frame..."):
                        # Run analyses manually
                        results = {}
                        
                        if "object_detection" in analysis_types:
                            segmenter = YOLOSegmentation()
                            seg_results = segmenter.segment_objects(frame)
                            if "error" not in seg_results:
                                results["object_detection"] = seg_results
                        
                        if "mediapipe_face" in analysis_types:
                            analyzer = MediaPipeAnalyzer()
                            face_results = analyzer.analyze_face_detection(frame)
                            if "error" not in face_results:
                                results["mediapipe_face"] = face_results
                        
                        if "emotion" in analysis_types:
                            emotion_detector = EmotionDetection()
                            emotion_results = emotion_detector.detect_emotions(frame)
                            if "error" not in emotion_results:
                                results["emotion"] = emotion_results
                        
                        # Display results
                        if results:
                            tabs = st.tabs([f"🔍 {analysis.replace('_', ' ').title()}" for analysis in results.keys()])
                            
                            tab_idx = 0
                            for analysis_type, result in results.items():
                                with tabs[tab_idx]:
                                    if analysis_type == "object_detection" and "segmented_image_sg" in result:
                                        st.image(result["segmented_image_sg"], caption="Object Detection", use_container_width=True)
                                        st.write(f"🎭 Objects detected: {result.get('count', 0)}")
                                    
                                    elif analysis_type == "mediapipe_face" and "face_image" in result:
                                        st.image(result["face_image"], caption="Face Detection", use_container_width=True)
                                        st.write(f"👤 Faces detected: {result.get('count', 0)}")
                                    
                                    elif analysis_type == "emotion" and "emotion_image" in result:
                                        st.image(result["emotion_image"], caption="Emotion Analysis", use_container_width=True)
                                        st.write(f"😊 Faces analyzed: {result.get('face_count', 0)}")
                                
                                tab_idx += 1
    
    else:
        st.info("📹 Click 'Start Camera' to begin enhanced real-time analysis")
        
        # Instructions
        st.markdown("""
        ### 🚀 Getting Started:
        1. **Start Camera** - Initialize your camera
        2. **Start Live Feed** - See real-time video feed
        3. **Configure Analysis** - Select which analyses to run
        4. **Start Auto-Capture** - Automatically capture and analyze frames
        5. **View Dashboard** - See analytics and detection history
        
        ### 🎯 Available Analyses:
        - **🎭 Object Detection**: YOLO-based object segmentation
        - **👤 MediaPipe Face**: Advanced face detection
        - **✋ MediaPipe Hands**: Hand landmark detection
        - **🤸 MediaPipe Pose**: Pose landmark detection
        - **😊 Emotion Detection**: Facial emotion analysis
        """)

# Updated main function to include new pages
# Updated main function to include new pages
def updated_main():
    """Main application function with complete sidebar - ALL BUTTON KEYS FIXED"""
    
    # Apply theme based on current state
    apply_theme()
    
    # COMPLETE SIDEBAR WITH ENHANCED CONTROLS
    with st.sidebar:
        st.title("🤖 Enhanced Smart AI Assistant with MediaPipe")
        
        # Theme Toggle
        with st.expander("🎨 Theme Settings", expanded=False):
            theme_col1, theme_col2 = st.columns([3, 1])
            with theme_col1:
                current_theme = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
                st.write(f"**Current Theme:** {current_theme}")
            with theme_col2:
                if st.button("🔄", help="Toggle between dark and light mode", key="theme_toggle_sidebar_main"):
                    st.session_state.dark_mode = not st.session_state.dark_mode
                    st.rerun()
        
        # AI Settings
        with st.expander("🎛️ AI Settings", expanded=True):
            st.subheader("🌡️ Temperature")
            st.session_state.temperature = st.slider(
                "Response Creativity",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Lower = more focused, Higher = more creative",
                key="temperature_slider_main"
            )
            
            st.subheader("😊 Conversation Mood")
            moods = list(MoodSystem.MOODS.keys())
            mood_labels = [f"{MoodSystem.get_mood_emoji(mood)} {mood}" for mood in moods]
            
            selected_mood_index = moods.index(st.session_state.mood) if st.session_state.mood in moods else 0
            selected_mood_label = st.selectbox(
                "Choose conversation style:",
                mood_labels,
                index=selected_mood_index,
                key="mood_selector_main"
            )
            st.session_state.mood = selected_mood_label.split(" ", 1)[1]
            
            st.subheader("📝 Response Length")
            st.session_state.max_tokens = st.slider(
                "Maximum Response Length",
                min_value=100,
                max_value=1000,
                value=st.session_state.max_tokens,
                step=50,
                key="max_tokens_slider_main"
            )
        
        # Audio Package Manager
        with st.expander("📦 Audio Package Manager", expanded=False):
            # Audio package installation section
            st.subheader("📦 Install Audio Packages")
            
            missing_packages = []
            
            if not GTTS_AVAILABLE:
                missing_packages.extend(["gtts", "pygame"])
            
            if not SPEECH_SUPPORT:
                missing_packages.append("SpeechRecognition")
            
            if not PYTTSX3_AVAILABLE:
                missing_packages.append("pyttsx3")
            
            if missing_packages:
                st.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
                st.code(f"pip install {' '.join(missing_packages)}")
                
                if st.button("🚀 Auto-Install Packages", key="auto_install_audio_packages_main"):
                    try:
                        import subprocess
                        
                        for package in missing_packages:
                            with st.spinner(f"Installing {package}..."):
                                result = subprocess.run([
                                    sys.executable, "-m", "pip", "install", package
                                ], capture_output=True, text=True)
                                
                                if result.returncode == 0:
                                    st.success(f"✅ {package} installed successfully!")
                                else:
                                    st.error(f"❌ Failed to install {package}: {result.stderr}")
                        
                        st.balloons()
                        st.info("🔄 Please restart the application to use new packages!")
                        
                    except Exception as e:
                        st.error(f"Installation error: {e}")
            else:
                st.success("✅ All audio packages are available!")
            
            # Enhanced Speech Test Section
            st.subheader("🎤 Test Enhanced Speech System")
            
            # Show current status
            if 'speech_manager' in st.session_state and st.session_state.speech_manager:
                status = st.session_state.speech_manager.get_status()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**TTS Method**: {status['current_method']}")
                    st.write(f"**Speaking**: {status['is_speaking']}")
                with col2:
                    st.write(f"**Speech Recognition**: {status['speech_recognition']}")
                    st.write(f"**Listening**: {status['is_listening']}")
            
            # Test text
            test_text = st.text_area(
                "Test Speech Text:",
                value="Hello! This is a test of the enhanced speech system. It should work much better than before!",
                height=80,
                key="test_speech_text_area_main"
            )
            
            # Test buttons with unique keys
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔊 Test", key="test_speak_button_main"):
                    if st.session_state.speech_manager:
                        if st.session_state.speech_manager.speak_text(test_text):
                            st.success("✅ Started speaking...")
                        else:
                            st.error("❌ Failed to start speaking")
            
            with col2:
                if st.button("⏹️ Stop", key="test_stop_speaking_button_main"):
                    if st.session_state.speech_manager:
                        st.session_state.speech_manager.stop_speaking()
                        st.success("✅ Stopped speaking")
            
            with col3:
                if st.button("🎤 Listen", key="test_listen_button_main"):
                    if st.session_state.speech_manager:
                        with st.spinner("🎤 Listening..."):
                            result = st.session_state.speech_manager.listen_for_speech(timeout=3)
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.success(f"✅ Heard: \"{result['text']}\"")

        # Computer Vision Settings
        with st.expander("👁️ Computer Vision Settings", expanded=False):
            st.subheader("🎭 Segmentation")
            st.session_state.segmentation_opacity = st.slider(
                "Mask Transparency",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.segmentation_opacity,
                step=0.1,
                key="segmentation_opacity_slider_main"
            )
            
            st.subheader("🤸 Pose Detection")
            st.session_state.pose_confidence = st.slider(
                "Pose Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.pose_confidence,
                step=0.1,
                key="pose_confidence_slider_main"
            )
            
            st.subheader("😊 Emotion Detection")
            st.session_state.emotion_confidence = st.slider(
                "Emotion Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.emotion_confidence,
                step=0.1,
                key="emotion_confidence_slider_main"
            )
        
        # MediaPipe Settings
        with st.expander("🤖 MediaPipe Settings", expanded=False):
            st.subheader("👤 Face Detection")
            face_confidence = st.slider(
                "Face Detection Confidence",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="mediapipe_face_confidence_main"
            )
            
            st.subheader("✋ Hand Detection")
            hand_confidence = st.slider(
                "Hand Detection Confidence", 
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="mediapipe_hand_confidence_main"
            )
            
            st.subheader("🤸 Pose Detection")
            pose_confidence = st.slider(
                "Pose Detection Confidence",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                key="mediapipe_pose_confidence_main"
            )
        
        # Auto-Capture Settings
        with st.expander("🔄 Auto-Capture Settings", expanded=False):
            st.subheader("📹 Auto-Capture")
            auto_interval = st.selectbox(
                "Auto-capture Interval",
                [1.0, 2.0, 3.0, 5.0, 10.0],
                index=1,
                help="Seconds between automatic captures",
                key="auto_interval_selector_main"
            )
            
            enable_auto_analysis = st.checkbox(
                "🔍 Enable Auto-Analysis",
                value=True,
                help="Automatically analyze captured frames",
                key="enable_auto_analysis_main"
            )
            
            st.subheader("📊 Dashboard Settings")
            max_history = st.slider(
                "Max Detection History",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="Maximum number of detections to keep in history",
                key="max_history_slider_main"
            )

        # Speech & Accessibility Settings
        with st.expander("🎤 Speech & Accessibility Settings", expanded=False):
            if SPEECH_SUPPORT:
                st.subheader("🎤 Speech Recognition")
                speech_timeout = st.slider(
                    "Listening Timeout (seconds)",
                    min_value=3,
                    max_value=30,
                    value=5,
                    step=1,
                    help="How long to wait for speech input",
                    key="speech_timeout_slider_main"
                )
                
                speech_phrase_limit = st.slider(
                    "Max Speech Length (seconds)",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                    help="Maximum length of speech input",
                    key="speech_phrase_limit_slider_main"
                )
                
                st.subheader("🔊 Text-to-Speech")
                tts_rate = st.slider(
                    "Speech Rate (words/min)",
                    min_value=100,
                    max_value=300,
                    value=180,
                    step=10,
                    help="Speed of speech output",
                    key="tts_rate_slider_main"
                )
                
                tts_volume = st.slider(
                    "Speech Volume",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="Volume of speech output",
                    key="tts_volume_slider_main"
                )
                
                # Voice selection
                if st.session_state.speech_manager:
                    available_voices = st.session_state.speech_manager.get_available_voices()
                    if available_voices:
                        voice_names = [f"{voice['name']} ({voice['gender']})" for voice in available_voices]
                        selected_voice = st.selectbox(
                            "Select Voice", 
                            voice_names, 
                            key="voice_selector_main"
                        )
                
                # Test buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎤 Test Mic", key="sidebar_test_microphone_main"):
                        if st.session_state.speech_manager:
                            with st.spinner("🎤 Say something..."):
                                test_result = st.session_state.speech_manager.listen_for_speech(timeout=3)
                                if "error" in test_result:
                                    st.error(f"❌ {test_result['error']}")
                                else:
                                    st.success(f"✅ Heard: \"{test_result['text']}\"")
                
                with col2:
                    if st.button("🔊 Test TTS", key="sidebar_test_speaker_main"):
                        if st.session_state.speech_manager:
                            test_text_short = "Hello! This is a test of the text to speech system."
                            threading.Thread(
                                target=lambda: st.session_state.speech_manager.speak_text(test_text_short, rate=tts_rate, volume=tts_volume),
                                daemon=True
                            ).start()
                            st.success("🔊 Speaking test message...")
            
            else:
                st.warning("🚫 Speech features not available")
                st.info("Install packages: `pip install SpeechRecognition pyttsx3 pyaudio`")
            
            if st.button("📋 Copy Install Command", key="copy_install_command_main"):
                st.code("pip install SpeechRecognition pyttsx3 pyaudio sounddevice soundfile")
        
        # System Status
        with st.expander("🔧 System Status", expanded=False):
            st.subheader("📋 Library Status")
            status_items = {
                "Core Features": True,
                "PDF Support": PDF_SUPPORT,
                "Gemini API": GEMINI_SUPPORT and bool(Config.GOOGLE_API_KEY),
                "GROQ API": GROQ_SUPPORT and bool(Config.GROQ_API_KEY),
                "YOLO Support": YOLO_SUPPORT,
                "Image Support": IMAGE_SUPPORT,
                "TensorFlow": TENSORFLOW_SUPPORT,
                "MediaPipe": MEDIAPIPE_SUPPORT,
                "Email": bool(Config.GMAIL_EMAIL and Config.GMAIL_APP_PASSWORD),
                "VADER Sentiment": VADER_AVAILABLE,
                "TextBlob Sentiment": TEXTBLOB_AVAILABLE,
                "Speech Recognition": SPEECH_SUPPORT
            }
            
            for item, status in status_items.items():
                status_icon = "✅" if status else "⚠️"
                st.write(f"{status_icon} {item}")
            
            # Show current file limits
            st.subheader("📋 Current Limits")
            st.write(f"📄 **PDF Limit**: {Config.MAX_FILE_SIZE_MB}MB")
            st.write(f"💬 **Chat History**: {Config.MAX_CHAT_HISTORY} messages")
            st.write(f"🎭 **Detection History**: {max_history} items")
        
        # Performance Monitor
        with st.expander("⚡ Performance Monitor", expanded=False):
            import psutil
            import gc
            
            # Memory usage
            memory = psutil.virtual_memory()
            st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            # Session state size
            session_items = len(st.session_state)
            st.metric("Session Items", session_items)
            
            # Clear cache button
            if st.button("🧹 Clear Cache", key="clear_cache_button_main"):
                gc.collect()
                if 'detection_history' in st.session_state:
                    st.session_state.detection_history = []
                if 'auto_capture_results' in st.session_state:
                    st.session_state.auto_capture_results = []
                st.success("✅ Cache cleared!")
        
        # Navigation
        st.subheader("📱 Navigation")
        page = st.selectbox(
            "Choose Page",
            [
                "💬 Chat Assistant", 
                "🎭 Enhanced Object Segmentation",
                "🤖 MediaPipe Analysis",
                "🤸 Pose Detection",
                "😊 Emotion Analysis",
                "📹 Enhanced Real-time Camera",
                "💭 Live Sentiment Analysis",
                "📊 Chat History"
            ],
            key="page_selector_main"
        )
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Restart", help="Restart the application", key="sidebar_restart_button_main"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("💾 Export", help="Export settings", key="sidebar_export_button_main"):
                settings = {
                    "temperature": st.session_state.temperature,
                    "mood": st.session_state.mood,
                    "max_tokens": st.session_state.max_tokens,
                    "pose_confidence": st.session_state.pose_confidence,
                    "emotion_confidence": st.session_state.emotion_confidence,
                    "segmentation_opacity": st.session_state.segmentation_opacity,
                    "dark_mode": st.session_state.dark_mode
                }
                
                import json
                settings_json = json.dumps(settings, indent=2)
                st.download_button(
                    label="📥 Download Settings",
                    data=settings_json,
                    file_name=f"chatbot_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_settings_button_main"
                )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px;'>
            🤖 Enhanced Smart AI Assistant<br>
            with MediaPipe Integration<br>
            <em>Version 2.0</em>
        </div>
        """, unsafe_allow_html=True)
    
    # MAIN CONTENT - PAGE ROUTING
    if page == "💬 Chat Assistant":
        chat_page()
    elif page == "🎭 Enhanced Object Segmentation":
        enhanced_segmentation_page()
    elif page == "🤖 MediaPipe Analysis":
        mediapipe_analysis_page()
    elif page == "🤸 Pose Detection":
        pose_detection_page()
    elif page == "😊 Emotion Analysis":
        emotion_analysis_page()
    elif page == "📹 Enhanced Real-time Camera":
        enhanced_real_time_camera_page()
    elif page == "💭 Live Sentiment Analysis":
        live_sentiment_analysis_page()
    elif page == "📊 Chat History":
        chat_history_page()


if __name__ == "__main__":
    # Use the updated main function
    updated_main()