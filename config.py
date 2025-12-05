"""
Configuration module for YouTube Script Generator
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration settings"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Paths
    DATA_DIR = "Data/processed"  # Legacy support
    SHORT_FORM_DATA_DIR = "Data/SHORT-FORM"
    LONG_FORM_DATA_DIR = "Data/LONG-FORM"
    USER_SCRIPTS_DIR = "user_scripts"
    OUTPUT_DIR = "output"
    
    # Application Settings
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    NUM_TRANSCRIPTS_TO_LOAD = int(os.getenv('NUM_TRANSCRIPTS_TO_LOAD', '25'))
    MAX_SCRIPT_LENGTH_CHARS = int(os.getenv('MAX_SCRIPT_LENGTH_CHARS', '20000'))
    SPEECH_WPM = int(os.getenv('SPEECH_WPM', '140'))  # average Hindi speaking pace
    SHORT_FORM_WPS = float(os.getenv('SHORT_FORM_WPS', '2.5'))  # words per second for short-form content
    
    # Thumbnail / Hugging Face Inference API (free tier works without token; token improves reliability)
    # Prefer widely available free endpoints; change via env if needed
    HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'black-forest-labs/FLUX.1-schnell')
    HF_MODEL_FALLBACK = os.getenv('HF_MODEL_FALLBACK', 'stabilityai/sd-turbo')
    HF_API_TOKEN = os.getenv('HF_API_TOKEN')  # optional
    THUMBNAIL_DIR = os.getenv('THUMBNAIL_DIR', 'thumbnails')
    
    # Default Script Parameters
    DEFAULT_TONE = os.getenv('DEFAULT_TONE', 'friendly_and_informative')
    DEFAULT_TARGET_AUDIENCE = os.getenv('DEFAULT_TARGET_AUDIENCE', 'tech_enthusiasts')
    DEFAULT_LENGTH_MINUTES = int(os.getenv('DEFAULT_LENGTH_MINUTES', '10'))
    
    # Valid Tones
    VALID_TONES = [
        'friendly_and_informative',
        'enthusiastic_and_energetic', 
        'professional_and_formal',
        'casual_and_conversational',
        'dramatic_and_engaging',
        'technical_and_detailed',
        'humorous_and_entertaining'
    ]
    
    # Valid Target Audiences
    VALID_AUDIENCES = [
        'tech_enthusiasts',
        'general_audience',
        'beginners',
        'professionals',
        'students',
        'gamers',
        'content_creators'
    ]
    
    # Supported Length Categories
    LENGTH_CATEGORIES = {
        'short': (2, 5),      # 2-5 minutes
        'medium': (5, 12),    # 5-12 minutes  
        'long': (12, 20),     # 12-20 minutes
        'very_long': (20, 40) # 20-40 minutes
    }
    
    # YouTube Creator Mapping (based on actual transcript data)
    YOUTUBE_CREATORS = {
        'Trakin Tech': {
            'style': 'tech_reviewer_indian',
            'language_mix': 'hinglish_heavy',
            'tone_preference': 'enthusiastic_and_energetic',
            'specialization': 'smartphone_reviews'
        },
        'TechBar': {
            'style': 'tech_reviewer_detailed', 
            'language_mix': 'hinglish_balanced',
            'tone_preference': 'friendly_and_informative',
            'specialization': 'comprehensive_tech_reviews'
        }
    }
    
    # Language Processing Settings
    MIN_TRANSCRIPT_WORDS = 100
    MAX_CONTEXT_WINDOW_GPT = 15000  # Leave room for generated content
    
    @classmethod
    def validate_config(cls):
        """Validate configuration on startup"""
        errors = []
        
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
        
        # Check if at least one data directory exists (legacy or new structure)
        if not os.path.exists(cls.DATA_DIR) and not os.path.exists(cls.SHORT_FORM_DATA_DIR) and not os.path.exists(cls.LONG_FORM_DATA_DIR):
            errors.append(f"None of the data directories exist: {cls.DATA_DIR}, {cls.SHORT_FORM_DATA_DIR}, or {cls.LONG_FORM_DATA_DIR}")
            
        # Create user scripts directory if it doesn't exist
        if not os.path.exists(cls.USER_SCRIPTS_DIR):
            os.makedirs(cls.USER_SCRIPTS_DIR, exist_ok=True)
            
        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
        
        return True

