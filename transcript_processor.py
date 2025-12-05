"""
Transcript Processor Module
Handles loading, preprocessing, and analysis of YouTube transcripts
"""

import json
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import nltk
import re
from collections import Counter
from langdetect import detect, DetectorFactory
import warnings
warnings.filterwarnings('ignore')

# Set seed for language detection consistency
DetectorFactory.seed = 0

@dataclass
class TranscriptMetadata:
    """Metadata for a YouTube transcript"""
    video_id: str
    title: str
    uploader: str
    duration: int
    view_count: int
    transcript_length: int
    word_count: int
    language_mix: float  # Ratio of English to Hindi words
    avg_segment_duration: float

@dataclass
class ProcessedTranscript:
    """Processed transcript with analysis"""
    metadata: TranscriptMetadata
    segments: List[Dict]
    clean_text: str
    language_breakdown: Dict[str, int]
    keywords: List[str]
    tone_markers: Dict[str, float]
    creator_style: Dict[str, any]

class TranscriptProcessor:
    """Main transcript processing class"""
    
    def __init__(self, data_dir: str = None, content_format: str = "short-form", genre: str = None):
        """
        Initialize transcript processor
        
        Args:
            data_dir: Legacy data directory path (optional)
            content_format: "short-form" or "long-form"
            genre: Genre name for long-form content (e.g., "Education", "Comedy")
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.content_format = content_format
        self.genre = genre
        self.processed_transcripts = []
        
        # Download required NLTK data
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def load_all_transcripts(self) -> List[ProcessedTranscript]:
        """Load and process all available transcripts based on content format"""
        from config import Config
        
        transcript_files = []
        
        # Determine which directory to load from
        if self.content_format == "short-form":
            # Load from SHORT-FORM directory
            short_form_dir = Path(Config.SHORT_FORM_DATA_DIR)
            if short_form_dir.exists():
                transcript_files = list(short_form_dir.glob("*_transcript.json"))
                print(f"Loading Short-form transcripts from {short_form_dir}")
            else:
                print(f"Warning: Short-form directory {short_form_dir} not found")
                return []
                
        elif self.content_format == "long-form":
            # Load from genre-specific directory
            if not self.genre:
                print("Error: Genre must be specified for long-form content")
                return []
            
            genre_dir = Path(Config.LONG_FORM_DATA_DIR) / self.genre
            if genre_dir.exists():
                transcript_files = list(genre_dir.glob("*_transcript.json"))
                print(f"Loading Long-form transcripts from {genre_dir}")
            else:
                print(f"Warning: Genre directory {genre_dir} not found")
                return []
        else:
            # Legacy mode: use provided data_dir
            if self.data_dir and self.data_dir.exists():
                transcript_files = list(self.data_dir.glob("*_transcript.json"))
                print(f"Loading transcripts from legacy directory {self.data_dir}")
            else:
                print(f"Warning: Data directory {self.data_dir} not found")
                return []
        
        print(f"Found {len(transcript_files)} transcript files")
        
        # Load all transcripts (remove limit for better training)
        for file_path in transcript_files:
            try:
                transcript = self.load_single_transcript(file_path)
                if transcript:
                    self.processed_transcripts.append(transcript)
                    print(f"[OK] Processed: {transcript.metadata.title}")
            except Exception as e:
                print(f"[ERROR] Error processing {file_path}: {e}")
        
        print(f"Successfully processed {len(self.processed_transcripts)} transcripts")
        return self.processed_transcripts
    
    @staticmethod
    def get_available_genres() -> List[str]:
        """Get list of available genres from LONG-FORM directory"""
        from config import Config
        
        genres_dir = Path(Config.LONG_FORM_DATA_DIR)
        if not genres_dir.exists():
            return []
        
        # Get all subdirectories that contain transcript files
        genres = []
        for genre_path in genres_dir.iterdir():
            if genre_path.is_dir():
                # Check if directory contains transcript files
                transcript_files = list(genre_path.glob("*_transcript.json"))
                if transcript_files:
                    genres.append(genre_path.name)
        
        return sorted(genres)
    
    def load_user_scripts(self, creator_id: str) -> List[ProcessedTranscript]:
        """Load user-uploaded scripts for a specific creator"""
        from config import Config
        
        user_scripts_dir = Path(Config.USER_SCRIPTS_DIR) / creator_id
        if not user_scripts_dir.exists():
            return []
        
        transcript_files = list(user_scripts_dir.glob("*_transcript.json"))
        print(f"Found {len(transcript_files)} user scripts for creator: {creator_id}")
        
        user_transcripts = []
        for file_path in transcript_files:
            try:
                transcript = self.load_single_transcript(file_path)
                if transcript:
                    user_transcripts.append(transcript)
                    print(f"[OK] Loaded user script: {transcript.metadata.title}")
            except Exception as e:
                print(f"[ERROR] Error loading user script {file_path}: {e}")
        
        return user_transcripts
    
    def load_single_transcript(self, file_path: Path) -> Optional[ProcessedTranscript]:
        """Load and process a single transcript file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract metadata
            metadata = self._extract_metadata(data)
            
            # Process transcript segments
            segments = data.get('transcript', [])
            
            # Skip very short transcripts
            if len(segments) < 10:
                return None
            
            # Extract and analyze text
            clean_text = self._extract_clean_text(segments)
            language_breakdown = self._analyze_language_mix(clean_text)
            keywords = self._extract_keywords(clean_text)
            tone_markers = self._analyze_tone_markers(clean_text)
            creator_style = self._analyze_creator_style(metadata, clean_text, segments)
            
            return ProcessedTranscript(
                metadata=metadata,
                segments=segments,
                clean_text=clean_text,
                language_breakdown=language_breakdown,
                keywords=keywords,
                tone_markers=tone_markers,
                creator_style=creator_style
            )
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_metadata(self, data: Dict) -> TranscriptMetadata:
        """Extract metadata from transcript data"""
        video_info = data.get('metadata', {})
        segments = data.get('transcript', [])
        
        # Calculate derived metrics
        total_words = sum(len(segment.get('text', '').split()) for segment in segments)
        avg_duration = sum(segment.get('duration', 0) for segment in segments) / len(segments) if segments else 0
        
        return TranscriptMetadata(
            video_id=data.get('video_id', ''),
            title=video_info.get('title', ''),
            uploader=video_info.get('uploader', ''),
            duration=video_info.get('duration', 0),
            view_count=video_info.get('view_count', 0),
            transcript_length=len(segments),
            word_count=total_words,
            language_mix=0,  # Will be calculated later
            avg_segment_duration=avg_duration
        )
    
    def _extract_clean_text(self, segments: List[Dict]) -> str:
        """Extract clean transcript text"""
        text_parts = []
        for segment in segments:
            text = segment.get('text', '').strip()
            if text and not self._is_music_or_sound(text):
                text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _is_music_or_sound(self, text: str) -> bool:
        """Check if segment contains music notation"""
        music_indicators = ['[संगीत]', '[Music]', '[musik]', '[music]', '♪', '♫']
        return any(indicator in text for indicator in music_indicators)
    
    def _analyze_language_mix(self, text: str) -> Dict[str, int]:
        """Analyze language mix in the text"""
        words = text.split()
        
        hindi_words = 0
        english_words = 0
        mixed_words = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
                
            # Simple detection based on script and character sets
            if re.search(r'[\u0900-\u097F]', word):  # Hindi Devanagari
                english_chars = len(re.findall(r'[a-zA-Z]', word))
                if english_chars > 2:
                    mixed_words += 1
                else:
                    hindi_words += 1
            elif re.search(r'[a-zA-Z]', word):
                english_words += 1
        
        total = hindi_words + english_words + mixed_words
        if total == 0:
            return {'hindi': 0, 'english': 0, 'mixed': 0, 'total': 0}
        
        return {
            'hindi': hindi_words,
            'english': english_words, 
            'mixed': mixed_words,
            'total': total,
            'hindi_ratio': hindi_words / total,
            'english_ratio': english_words / total,
            'mixed_ratio': mixed_words / total
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove common English/Hindi stopwords
        english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        hindi_stopwords = {'और', 'कि', 'या', 'लेकिन', 'में', 'पर', 'के', 'से', 'तक', 'है', 'हैं', 'था', 'थी', 'थे', 'हो', 'होगा', 'होगी', 'होते', 'मैं', 'तुम', 'वह', 'हम', 'आप', 'यह', 'वे'}
        
        filtered_words = [w for w in words if w not in english_stopwords and w not in hindi_stopwords and len(w) > 2]
        
        # Get most frequent words
        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(20)]
    
    def _analyze_tone_markers(self, text: str) -> Dict[str, float]:
        """Analyze tone markers in the text"""
        # Simple tone analysis based on keywords and patterns
        enthusiastic_words = ['amazing', 'fantastic', 'awesome', 'great', 'excellent', 'incredible', 'बहुत', 'अच्छा', 'शानदार']
        technical_words = ['specifications', 'features', 'performance', 'benchmark', 'technology', 'स्पेक्स', 'फीचर्स', 'परफॉर्मेंस']
        friendly_words = ['friends', 'guys', 'bhai', 'दोस्तों', 'भाई', 'yaar', 'यार']
        
        text_lower = text.lower()
        
        enthusiasm_score = sum(1 for word in enthusiastic_words if word in text_lower) / len(text.split()) * 100
        technical_score = sum(1 for word in technical_words if word in text_lower) / len(text.split()) * 100  
        friendliness_score = sum(1 for word in friendly_words if word in text_lower) / len(text.split()) * 100
        
        return {
            'enthusiasm': min(enthusiasm_score, 10.0),
            'technical_depth': min(technical_score, 10.0),
            'friendliness': min(friendliness_score, 10.0)
        }
    
    def _analyze_creator_style(self, metadata: TranscriptMetadata, text: str, segments: List[Dict]) -> Dict[str, any]:
        """Analyze creator-specific style patterns"""
        uploader = metadata.uploader
        
        # Analyze speaking patterns
        segments_with_text = [s for s in segments if s.get('text', '').strip() and not self._is_music_or_sound(s.get('text', ''))]
        
        avg_segment_duration = sum(s.get('duration', 0) for s in segments_with_text) / len(segments_with_text) if segments_with_text else 0
        
        # Analyze common phrases and style markers
        style_markers = []
        
        if 'trakin' in uploader.lower() or 'trakin tech' in uploader.lower():
            style_markers.extend(['tech_review', 'detailed_analysis', 'phone_comparison', 'price_analysis'])
        elif 'techbar' in uploader.lower():
            style_markers.extend(['comprehensive_review', 'technical_depth', 'long_form_content'])
        
        return {
            'creator_name': uploader,
            'style_markers': style_markers,
            'avg_segment_duration': avg_segment_duration,
            'unique_phrases': self._extract_unique_phrases(text)[:10],
            'speaking_pace': metadata.word_count / (metadata.duration / 60) if metadata.duration > 0 else 0  # words per minute
        }
    
    def _extract_unique_phrases(self, text: str, min_length: int = 3) -> List[str]:
        """Extract unique phrases that could be creator style markers"""
        sentences = text.split('.')
        phrases = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) >= min_length:
                phrases.append(' '.join(words))
        
        # Get most common phrases (simplified)
        phrase_count = Counter(phrases)
        return [phrase for phrase, count in phrase_count.most_common(20)]
    
    def get_creator_summary(self) -> Dict[str, Dict]:
        """Get summary of all creators and their styles"""
        creator_summaries = {}
        
        for transcript in self.processed_transcripts:
            creator = transcript.metadata.uploader
            style_data = transcript.creator_style
            
            if creator not in creator_summaries:
                creator_summaries[creator] = {
                    'video_count': 0,
                    'total_duration': 0,
                    'total_words': 0,
                    'avg_tone': {'enthusiasm': 0, 'technical_depth': 0, 'friendliness': 0},
                    'language_mix': {'hindi_ratio': 0, 'english_ratio': 0, 'mixed_ratio': 0},
                    'common_keywords': [],
                    'style_markers': set()
                }
            
            summary = creator_summaries[creator]
            summary['video_count'] += 1
            summary['total_duration'] += transcript.metadata.duration
            summary['total_words'] += transcript.metadata.word_count
            
            # Average tone markers
            for tone in ['enthusiasm', 'technical_depth', 'friendliness']:
                summary['avg_tone'][tone] += transcript.tone_markers[tone]
            
            # Average language mix
            for lang in ['hindi_ratio', 'english_ratio', 'mixed_ratio']:
                summary['language_mix'][lang] += transcript.language_breakdown[lang]
            
            # Merge keywords and style markers
            summary['common_keywords'].extend(transcript.keywords)
            summary['style_markers'].update(style_data['style_markers'])
        
        # Normalize averages
        for creator, summary in creator_summaries.items():
            count = summary['video_count']
            
            # Normalize tone averages
            for tone in summary['avg_tone']:
                summary['avg_tone'][tone] /= count
            
            # Normalize language mix averages  
            for lang in summary['language_mix']:
                summary['language_mix'][lang] /= count
            
            # Get most common keywords
            keyword_count = Counter(summary['common_keywords'])
            summary['common_keywords'] = [word for word, count in keyword_count.most_common(15)]
            
            # Convert set to list for JSON serialization
            summary['style_markers'] = list(summary['style_markers'])
        
        return creator_summaries
    
    def filter_transcripts_by_creator(self, creator_name: str) -> List[ProcessedTranscript]:
        """Filter transcripts by a specific creator"""
        return [t for t in self.processed_transcripts if t.metadata.uploader.lower() == creator_name.lower()]
    
    def get_training_dataset(self) -> List[Dict]:
        """Create training dataset for Gemini model"""
        training_data = []
        
        for transcript in self.processed_transcripts:
            # Create training examples
            training_entry = {
                'input': {
                    'creator_style': transcript.creator_style,
                    'tone_markers': transcript.tone_markers,
                    'language_mix': transcript.language_breakdown,
                    'metadata': {
                        'title': transcript.metadata.title,
                        'duration_minutes': transcript.metadata.duration // 60,
                        'uploader': transcript.metadata.uploader
                    }
                },
                'output': transcript.clean_text[:5000]  # Limit training text length
            }
            training_data.append(training_entry)
        
        return training_data

