"""
Script Generator Module
Uses Google Gemini API to generate YouTube scripts based on transcript analysis
"""

import google.generativeai as genai
import os

# Ensure Python and gRPC/OpenSSL use a valid CA bundle. This helps when the
# local environment or corporate proxy provides a custom/root cert that is
# not present in the default OpenSSL store. We prefer certifi when available
# and set environment variables used by requests, OpenSSL, and gRPC.
try:
    import certifi
    ca_path = certifi.where()
    # Standard env var for many Python TLS stacks
    os.environ.setdefault("SSL_CERT_FILE", ca_path)
    # requests (and some libraries) will honor this
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_path)
    # gRPC (C-core) can use this path when GRPC_DEFAULT_SSL_ROOTS_FILE_PATH is set
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", ca_path)
except Exception:
    # If certifi isn't installed, we leave system defaults in place. Users
    # should `pip install certifi` in their venv if they see certificate errors.
    pass
from typing import Dict, List, Optional, Tuple
import json
import tiktoken
from typing import Dict
import time
from config import Config
from transcript_processor import ProcessedTranscript

# Import RAG system
try:
    from rag_system import RAGSystem
    RAG_AVAILABLE = True
except (ImportError, ValueError) as e:
    RAG_AVAILABLE = False
    print(f"[WARN] RAG system not available: {e}")
    print("[WARN] Install chromadb and onnxruntime for RAG features: pip install chromadb onnxruntime")

class ScriptGenerator:
    """Main script generation class using Gemini API"""
    
    def __init__(self):
        """Initialize the script generator with Gemini configuration"""
        self.api_key = Config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        
        # Initialize Gemini models
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,  # Safer default; per-call overrides based on target length
        }
        
        # Safety settings to reduce blocking
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            safety_settings=self.safety_settings
        )
        
        # Training data cache
        self.training_context = {}
        self.creator_styles = {}
        
        # RAG system (initialized during training)
        self.rag_system = None
        if RAG_AVAILABLE:
            try:
                self.rag_system = RAGSystem()
            except Exception as e:
                print(f"[WARN] Failed to initialize RAG system: {e}")
        
        print("+ Script Generator initialized with Gemini API")
    
    def train_on_transcripts(self, processed_transcripts: List[ProcessedTranscript]) -> Dict[str, any]:
        """Train/initialize the system with transcript data"""
        print("Training script generator on transcript data...")
        
        # Analyze creator styles
        self.creator_styles = self._analyze_creator_styles(processed_transcripts)
        
        # Create training prompts and examples
        self.training_context = self._create_training_context(processed_transcripts)
        
        # Initialize RAG system with transcripts
        if self.rag_system:
            try:
                self.rag_system.initialize(processed_transcripts)
                print("[OK] RAG system initialized with transcripts")
            except Exception as e:
                print(f"[WARN] RAG initialization failed: {e}")
        
        # Test generation with sample
        sample_result = self._test_generation()
        
        training_summary = {
            'total_transcripts': len(processed_transcripts),
            'creators_analyzed': len(self.creator_styles),
            'training_examples': len(self.training_context.get('examples', [])),
            'sample_generation_success': sample_result['success'],
            'rag_initialized': self.rag_system is not None and self.rag_system._initialized
        }
        
        print(f"[OK] Training completed: {training_summary}")
        return training_summary
    
    def _analyze_creator_styles(self, transcripts: List[ProcessedTranscript]) -> Dict[str, Dict]:
        """Analyze and extract unique styles from each creator with actual examples"""
        from collections import Counter
        import re
        
        creator_analysis = {}
        
        for transcript in transcripts:
            creator = transcript.metadata.uploader
            
            if creator not in creator_analysis:
                creator_analysis[creator] = {
                    'transcripts': [],
                    'transcript_examples': [],  # Store actual text samples
                    'common_patterns': set(),
                    'language_preferences': {'hindi_dominant': 0, 'english_dominant': 0, 'balanced': 0},
                    'tone_profile': {'enthusiasm': 0, 'technical_depth': 0, 'friendliness': 0},
                    'content_style': set(),
                    'intro_patterns': [],
                    'outro_patterns': [],
                    'unique_phrases': Counter(),  # Track unique phrases
                    'common_expressions': Counter(),  # Track common expressions
                    'transition_phrases': Counter(),  # Track transition words
                    'speaking_pace_samples': []  # Store speaking pace data
                }
            
            analysis = creator_analysis[creator]
            analysis['transcripts'].append(transcript)
            
            # Store actual transcript text samples (500-1000 chars from different parts)
            clean_text = transcript.clean_text
            if len(clean_text) > 500:
                # Store beginning sample (first 800 chars)
                sample_start = clean_text[:800]
                # Store middle sample if transcript is long enough
                if len(clean_text) > 2000:
                    mid_point = len(clean_text) // 2
                    sample_mid = clean_text[mid_point:mid_point+800]
                    analysis['transcript_examples'].append({
                        'type': 'middle',
                        'text': sample_mid,
                        'title': transcript.metadata.title[:60]  # Store title for reference
                    })
                analysis['transcript_examples'].append({
                    'type': 'start',
                    'text': sample_start,
                    'title': transcript.metadata.title[:60]
                })
            
            # Analyze language preferences
            lang_mix = transcript.language_breakdown
            if lang_mix['hindi_ratio'] > 0.6:
                analysis['language_preferences']['hindi_dominant'] += 1
            elif lang_mix['english_ratio'] > 0.6:
                analysis['language_preferences']['english_dominant'] += 1
            else:
                analysis['language_preferences']['balanced'] += 1
            
            # Analyze tone profile
            for tone in ['enthusiasm', 'technical_depth', 'friendliness']:
                analysis['tone_profile'][tone] += transcript.tone_markers[tone]
            
            # Extract style markers
            analysis['content_style'].update(transcript.creator_style.get('style_markers', []))
            
            # Capture more intro/outro patterns (10-15 segments each)
            intro_segments = transcript.segments[:15]  # First 15 segments
            outro_segments = transcript.segments[-15:]  # Last 15 segments
            
            # Extract intro patterns (combine consecutive segments for better context)
            intro_texts = []
            current_intro = []
            for s in intro_segments:
                text = s.get('text', '').strip()
                if text:
                    current_intro.append(text)
                    # Group into phrases of 2-3 segments
                    if len(current_intro) >= 2:
                        intro_phrase = ' '.join(current_intro)
                        if len(intro_phrase) > 20:  # Only meaningful phrases
                            intro_texts.append(intro_phrase)
                        current_intro = []
            if current_intro:
                intro_texts.append(' '.join(current_intro))
            analysis['intro_patterns'].extend(intro_texts)
            
            # Extract outro patterns
            outro_texts = []
            current_outro = []
            for s in outro_segments:
                text = s.get('text', '').strip()
                if text:
                    current_outro.append(text)
                    if len(current_outro) >= 2:
                        outro_phrase = ' '.join(current_outro)
                        if len(outro_phrase) > 20:
                            outro_texts.append(outro_phrase)
                        current_outro = []
            if current_outro:
                outro_texts.append(' '.join(current_outro))
            analysis['outro_patterns'].extend(outro_texts)
            
            # Extract unique phrases and expressions
            text_lower = clean_text.lower()
            
            # Common Hinglish expressions
            hinglish_patterns = [
                r'\b(दोस्तों|भाई|यार|सुनिए|देखिए|तो यहाँ पर|बात यह है|अब देखते हैं|मुख्य बात)\b',
                r'\b(dosto|bhai|yaar|sunye|dekhiye|to yahan par|baat yeh hai|ab dekhte hain|mukhya baat)\b',
            ]
            
            for pattern in hinglish_patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ''
                    if match:
                        analysis['common_expressions'][match] += 1
            
            # Extract transition phrases (common between sentences)
            sentences = re.split(r'[.!?।]\s+', clean_text)
            transition_patterns = [
                r'^(अब|तो|लेकिन|फिर|वैसे|सबसे पहले|इसके अलावा|अब बात करते हैं|अब आते हैं)',
                r'^(now|so|but|then|also|first|apart from|lets talk about|coming to)',
            ]
            for sentence in sentences:
                sentence_stripped = sentence.strip()[:50]  # First 50 chars
                for pattern in transition_patterns:
                    if re.search(pattern, sentence_stripped, re.IGNORECASE):
                        transition_match = re.match(pattern, sentence_stripped, re.IGNORECASE)
                        if transition_match:
                            phrase = transition_match.group(0)
                            analysis['transition_phrases'][phrase.lower()] += 1
            
            # Store speaking pace (words per minute)
            if transcript.metadata.duration > 0:
                wpm = (transcript.metadata.word_count / (transcript.metadata.duration / 60.0))
                analysis['speaking_pace_samples'].append(wpm)
        
        # Normalize creator analysis
        for creator, analysis in creator_analysis.items():
            transcript_count = len(analysis['transcripts'])
            
            # Normalize tone profile
            for tone in analysis['tone_profile']:
                analysis['tone_profile'][tone] /= transcript_count
            
            # Convert sets to lists for serialization
            analysis['common_patterns'] = list(analysis['common_patterns'])
            analysis['content_style'] = list(analysis['content_style'])
            
            # Deduplicate and limit intro patterns (keep top 15 most common/longest)
            intro_patterns_cleaned = []
            seen_intros = set()
            for intro in analysis['intro_patterns']:
                intro_lower = intro.lower().strip()
                if intro_lower and intro_lower not in seen_intros and len(intro) > 15:
                    seen_intros.add(intro_lower)
                    intro_patterns_cleaned.append(intro)
            # Keep up to 15, prioritizing longer ones
            intro_patterns_cleaned.sort(key=len, reverse=True)
            analysis['intro_patterns'] = intro_patterns_cleaned[:15]
            
            # Deduplicate and limit outro patterns
            outro_patterns_cleaned = []
            seen_outros = set()
            for outro in analysis['outro_patterns']:
                outro_lower = outro.lower().strip()
                if outro_lower and outro_lower not in seen_outros and len(outro) > 15:
                    seen_outros.add(outro_lower)
                    outro_patterns_cleaned.append(outro)
            outro_patterns_cleaned.sort(key=len, reverse=True)
            analysis['outro_patterns'] = outro_patterns_cleaned[:15]
            
            # Get top unique phrases and expressions (convert Counter to dict with top items)
            analysis['unique_phrases'] = dict(analysis['unique_phrases'].most_common(20))
            analysis['common_expressions'] = dict(analysis['common_expressions'].most_common(15))
            analysis['transition_phrases'] = dict(analysis['transition_phrases'].most_common(10))
            
            # Calculate average speaking pace
            if analysis['speaking_pace_samples']:
                analysis['avg_speaking_pace'] = sum(analysis['speaking_pace_samples']) / len(analysis['speaking_pace_samples'])
            else:
                analysis['avg_speaking_pace'] = 140  # Default WPM
            
            # Keep only transcript examples (limit to 10-12 examples total)
            if len(analysis['transcript_examples']) > 12:
                # Prioritize start samples, then diversify
                start_samples = [e for e in analysis['transcript_examples'] if e['type'] == 'start']
                other_samples = [e for e in analysis['transcript_examples'] if e['type'] != 'start']
                analysis['transcript_examples'] = start_samples[:8] + other_samples[:4]
        
        return creator_analysis
    
    def _create_training_context(self, transcripts: List[ProcessedTranscript]) -> Dict:
        """Create training context and examples for Gemini"""
        # Organize examples by creator and style
        examples_by_creator = {}
        
        for transcript in transcripts:
            creator = transcript.metadata.uploader
            
            if creator not in examples_by_creator:
                examples_by_creator[creator] = []
            
            # Create training example
            example = {
                'title': transcript.metadata.title,
                'creator': creator,
                'style_summary': {
                    'tone_markers': transcript.tone_markers,
                    'language_mix': transcript.language_breakdown,
                    'title_type': self._analyze_title_type(transcript.metadata.title),
                    'duration_category': self._categorize_duration(transcript.metadata.duration)
                },
                'sample_content': transcript.clean_text[:2000],  # First 2000 chars
                'key_patterns': transcript.keywords[:10]
            }
            
            examples_by_creator[creator].append(example)
        
        # Create training prompts
        training_prompts = self._generate_training_prompts(examples_by_creator)
        
        return {
            'examples_by_creator': examples_by_creator,
            'training_prompts': training_prompts,
            'style_guidelines': self._create_style_guidelines()
        }
    
    def _analyze_title_type(self, title: str) -> str:
        """Analyze the type of video based on title"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['review', 'रीव्यू', 'test', 'टेस्ट']):
            return 'review'
        elif any(word in title_lower for word in ['vs', 'versus', 'कंपेयर', 'comparison']):
            return 'comparison'
        elif any(word in title_lower for word in ['unboxing', 'अनबॉक्सिंग', 'first look']):
            return 'unboxing'
        elif any(word in title_lower for word in ['guide', 'गाइड', 'tips', 'how to']):
            return 'tutorial'
        else:
            return 'general'
    
    def _categorize_duration(self, duration_seconds: int) -> str:
        """Categorize video duration"""
        duration_minutes = duration_seconds // 60
        
        if duration_minutes <= 5:
            return 'short'
        
        return 'medium'
    
    def _generate_training_prompts(self, examples_by_creator: Dict) -> List[Dict]:
        """Generate context-aware training prompts"""
        prompts = []
        
        for creator, examples in examples_by_creator.items():
            if len(examples) > 0:
                # Create creator-specific prompt
                creator_prompt = self._create_creator_specific_prompt(creator, examples[0])
                prompts.append(creator_prompt)
        
        return prompts
    
    def _create_creator_specific_prompt(self, creator: str, example: Dict) -> Dict:
        """Create a prompt specific to a creator's style"""
        
        # Base system prompt
        base_prompt = f"""You are generating YouTube scripts in the style of {creator}, a popular Hinglish (Hindi + English) tech YouTuber. 

Creator Style Guidelines for {creator}:
"""
        
        # Add creator-specific guidelines based on analysis
        if 'trakin' in creator.lower():
            base_prompt += """
- Use enthusiastic and energetic tone
- Mix Hindi and English naturally (Hinglish)
- Focus on smartphone reviews and comparisons
- Include detailed specifications and pricing
- Use expressions like "दोस्तों" frequently
- Include call-to-action for likes and subscriptions
- Structure: Hook intro -> Product overview -> Detailed review -> Pricing -> Conclusion
"""
        elif 'techbar' in creator.lower():
            base_prompt += """
- Use friendly and informative tone
- Provide comprehensive technical analysis
- Longer form content structure
- Professional yet approachable language style
- Detailed comparisons and features breakdown
- Include real-world usage scenarios
"""
        else:
            base_prompt += """
- Maintain engaging tech content style
- Mix conversational Hindi and English
- Focus on technology reviews and guides
- Include practical insights and comparisons
"""
        
        base_prompt += f"""

Generate authentic YouTube scripts that:
1. Match {creator}'s speaking patterns and style
2. Use appropriate Hinglish mix
3. Include engaging elements like hooks, transitions, and CTAs
4. Are structured for {example['style_summary']['duration_category']}-length videos
5. Capture the creator's unique voice and personality

Remember: The script should sound like {creator} actually wrote and spoke it - authentic and natural."""

        return {
            'creator': creator,
            'type': 'creator_specific',
            'prompt': base_prompt,
            'example': example
        }
    
    def _create_style_guidelines(self) -> Dict[str, str]:
        """Create general style guidelines for script generation"""
        return {
            'hinglish_patterns': {
                'common_transitions': ['अब देखते हैं', 'तो यहाँ पर', 'बात यह है कि', 'अब मुख्य बात'],
                'engagement_phrases': ['दोस्तों', 'भाई', 'यार', 'आपको पता है', 'सुनने के लिए'],
                'technical_mix': ['इसके में फीचर्स हैं', 'स्पेक्स देखिए', 'परफॉर्मेंस बहुत अच्छा है']
            },
            'script_structure': {
                'hook': 'Start with engaging question or statement',
                'intro': 'Brief intro with channel branding',
                'main_content': 'Core review/content with logical flow',
                'cta': 'Call to action for engagement',
                'outro': 'Channel promotion and subscription reminder'
            },
            'content_types': {
                'review': 'Focus on detailed analysis, pros/cons, recommendations',
                'comparison': 'Side-by-side comparison with clear winner',
                'unboxing': 'Step-by-step unboxing with reactions',
                'tutorial': 'Clear instructions with visual cues',
                'general': 'Balanced informative and entertaining content'
            }
        }
    
    def _test_generation(self) -> Dict[str, any]:
        """Test generation capability with a sample prompt"""
        try:
            test_prompt = """Generate a 2-minute YouTube script intro for a smartphone review video in Hinglish style used by Indian tech YouTubers."""
            
            response = self.model.generate_content(
                test_prompt,
                generation_config=self.generation_config
            )
            
            return {
                'success': True,
                'sample_length': len(response.text) if response.text else 0,
                'response_preview': response.text[:200] if response.text else 'No content generated'
            }
            
        except Exception as e:
            print(f"Test generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_script(self, 
                       topic: str,
                       length_minutes: int = None,
                       length_seconds: int = None,
                       tone: str = None,
                       target_audience: str = None,
                       content_type: str = None,
                       content_format: str = "long-form",
                       style_mode: Optional[str] = None,
                       creator_style: Optional[str] = None,
                       outline: Optional[str] = None,
                       facts: Optional[str] = None,
                       language_mix: str = "Balanced",
                       force_hinglish_ascii: bool = True) -> Dict[str, any]:
        """Generate a YouTube script based on parameters"""
        
        # Determine content format and length
        if content_format == "short-form" and length_seconds:
            effective_length_seconds = length_seconds
            effective_length_minutes = length_seconds / 60.0
            # Auto-calculate word cap for short-form
            hard_word_cap = max(25, int(length_seconds * Config.SHORT_FORM_WPS))
            print(f"Generating Short-form script: {topic} ({length_seconds} sec, {tone}, {target_audience})")
        else:
            # Long-form or default
            effective_length_minutes = length_minutes if length_minutes else 10
            effective_length_seconds = effective_length_minutes * 60
            # Auto-calculate word cap for long-form
            hard_word_cap = max(150, int(effective_length_minutes * Config.SPEECH_WPM))
            print(f"Generating Long-form script: {topic} ({effective_length_minutes} min, {tone}, {target_audience})")
        
        try:
            start_time = time.time()
            
            # Build prompt reflecting the exact cap
            prompt = self._build_generation_prompt(
                topic, effective_length_minutes, effective_length_seconds, tone, target_audience,
                content_type, content_format, style_mode, creator_style, outline, facts,
                hard_word_cap, language_mix, force_hinglish_ascii=force_hinglish_ascii
            )

            # Estimate prompt size (rough: ~4 chars per token, conservative)
            prompt_size_estimate = len(prompt) // 4
            
            # Check if prompt is too large - if it exceeds ~500 tokens, we might hit limits
            # Reduce prompt by truncating optional parts if needed
            max_prompt_tokens = 500  # Conservative limit to ensure output room
            if prompt_size_estimate > max_prompt_tokens:
                print(f"Warning: Prompt is large ({prompt_size_estimate} estimated tokens). Reducing outline/facts...")
                # Truncate outline and facts if they're too long
                if outline and len(outline) > 200:
                    outline = outline[:200] + "... [truncated for token limit]"
                if facts and len(facts) > 300:
                    facts = facts[:300] + "... [truncated for token limit]"
                
                # Rebuild prompt with truncated content
                prompt = self._build_generation_prompt(
                    topic, effective_length_minutes, effective_length_seconds, tone, target_audience,
                    content_type, content_format, style_mode, creator_style, outline, facts,
                    hard_word_cap, language_mix, force_hinglish_ascii=force_hinglish_ascii
                )
                prompt_size_estimate = len(prompt) // 4
                print(f"After reduction: {prompt_size_estimate} estimated tokens")

            # Compute a safe per-call token cap based on desired word count (~1.3 tokens/word)
            approx_tokens = int(hard_word_cap * 1.3)  # rough tokens-per-word multiplier
            
            # CRITICAL: Ensure we have enough output tokens by checking prompt size
            # Gemini has limits on total request size. If prompt is large, we need more output tokens
            # The issue: if prompt uses 632 tokens and total is 887, only 255 output tokens are available
            # Solution: Request maximum output tokens (8192) to ensure we get enough room
            # For large prompts (>400 tokens), always use max output tokens
            if prompt_size_estimate > 400:
                # Large prompt - use maximum output tokens to ensure we get enough room
                max_output_tokens = 8192
                print(f"Large prompt detected ({prompt_size_estimate} tokens). Using max output tokens: {max_output_tokens}")
            else:
                # Smaller prompt - can use calculated amount with minimum
                min_output_tokens = 3072
                max_output_tokens = max(min_output_tokens, min(8192, approx_tokens + 3072))
            
            call_generation_config = {**self.generation_config, "max_output_tokens": max_output_tokens}
            # Increase creativity for cinematic short-form
            if content_format == "short-form":
                call_generation_config["temperature"] = 0.9
                call_generation_config["top_p"] = 0.9
            print(f"Requesting {max_output_tokens} output tokens for prompt of ~{prompt_size_estimate} tokens")
            
            # Try generation with full prompt first
            response = self.model.generate_content(
                prompt,
                generation_config=call_generation_config
            )
            
            generation_time = time.time() - start_time
            
            # IMMEDIATELY check for empty content with MAX_TOKENS - this is the critical issue
            script_text = self._extract_text_from_response(response)
            has_empty_content = not script_text or len(script_text.strip()) == 0
            
            # Check finish reason and usage metadata
            finish_reason = None
            prompt_token_count = None
            total_token_count = None
            
            if getattr(response, 'candidates', None) and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
            
            # Get actual token usage from response if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_token_count = getattr(usage, 'prompt_token_count', None)
                total_token_count = getattr(usage, 'total_token_count', None)
            
            # If we have empty content and MAX_TOKENS, we need to retry immediately with higher limit
            if has_empty_content and finish_reason == "MAX_TOKENS":
                # Calculate available output tokens from the failed attempt
                available_output_tokens = (total_token_count - prompt_token_count) if (total_token_count and prompt_token_count) else None
                
                print(f"Warning: Empty content with MAX_TOKENS. Prompt tokens: {prompt_token_count}, Total: {total_token_count}")
                if available_output_tokens:
                    print(f"Only {available_output_tokens} tokens available for output - prompt is too large!")
                print(f"Initial max_output_tokens was: {max_output_tokens}. Retrying with reduced prompt and maximum tokens...")
                
                # If available output tokens are very low (<500), the prompt is definitely too large
                # Try reducing the prompt significantly and retry
                reduced_outline = None
                reduced_facts = None
                if available_output_tokens and available_output_tokens < 500:
                    # Prompt is too large - reduce optional parts more aggressively
                    print("Prompt too large - reducing outline and facts more aggressively...")
                    if outline:
                        reduced_outline = outline[:100] + "... [reduced for token limit]"
                    if facts:
                        reduced_facts = facts[:150] + "... [reduced for token limit]"
                    
                    # Rebuild with reduced prompt
                    prompt = self._build_generation_prompt(
                        topic, effective_length_minutes, effective_length_seconds, tone, target_audience,
                        content_type, content_format, creator_style, reduced_outline, reduced_facts,
                        hard_word_cap, language_mix, force_hinglish_ascii=force_hinglish_ascii
                    )
                    print(f"Rebuilt prompt with reduced content. New size: ~{len(prompt) // 4} tokens")
                
                # Use maximum available tokens (8192 for gemini-2.5-flash)
                retry_max_output = 8192
                retry_config = {**self.generation_config, "max_output_tokens": retry_max_output}
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=retry_config
                )
                
                # Check again after retry
                script_text = self._extract_text_from_response(response)
                has_empty_content = not script_text or len(script_text.strip()) == 0
                
                if getattr(response, 'candidates', None) and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    prompt_token_count = getattr(usage, 'prompt_token_count', None)
                    total_token_count = getattr(usage, 'total_token_count', None)
                    available_output_tokens = (total_token_count - prompt_token_count) if (total_token_count and prompt_token_count) else None
                
                # If still empty after max retry with reduced prompt, prompt is definitely too large
                if has_empty_content and finish_reason == "MAX_TOKENS":
                    error_msg = 'Script generation failed: The prompt is too large for the API to generate content.\n\n'
                    if prompt_token_count:
                        error_msg += f'• Prompt uses {prompt_token_count} tokens\n'
                    if available_output_tokens:
                        error_msg += f'• Only {available_output_tokens} tokens available for output\n'
                    error_msg += '\n**Solutions:**\n'
                    error_msg += '1. Reduce the "Facts & Information" field - keep it under 300 characters\n'
                    error_msg += '2. Shorten the "Script Outline" - keep it under 200 characters\n'
                    error_msg += '3. Simplify the topic description\n'
                    error_msg += '4. Try generating a shorter script (reduce duration)\n'
                    
                    return {
                        'success': False,
                        'error': error_msg,
                        'metadata': {
                            'topic': topic,
                            'length_minutes': effective_length_minutes if content_format != "short-form" else None,
                            'length_seconds': effective_length_seconds if content_format == "short-form" else None,
                            'finish_reason': 'MAX_TOKENS',
                            'max_output_tokens': retry_max_output,
                            'prompt_tokens': prompt_token_count,
                            'output_tokens_available': available_output_tokens
                        }
                    }
            
            # Also check for MAX_TOKENS even if we got some content (might be truncated)
            if finish_reason == "MAX_TOKENS" and not has_empty_content:
                print(f"Warning: Response truncated due to MAX_TOKENS (had {max_output_tokens} limit)")
                # If we got content but hit limit, try once more with max tokens
                if max_output_tokens < 8192:
                    retry_max_output = 8192
                    retry_config = {**self.generation_config, "max_output_tokens": retry_max_output}
                    retry_response = self.model.generate_content(
                        prompt,
                        generation_config=retry_config
                    )
                    retry_text = self._extract_text_from_response(retry_response)
                    if retry_text and len(retry_text) > len(script_text):
                        # Got more content, use it
                        script_text = retry_text
                        response = retry_response
                        print(f"Retry with max tokens ({retry_max_output}) produced more content")
            
            # Check for safety issues
            if not response.candidates:
                # Try with a simpler, safer prompt
                print("First attempt blocked, trying with simpler prompt...")
                lang_desc = "English" if language_mix == "English" else "Hinglish (Hindi + English mix)"
                if content_format == "short-form":
                    duration_desc = f"{effective_length_seconds} seconds"
                    simple_prompt = (
                        f"Write a cinematic short-form reel VO about {topic} in {lang_desc}. "
                        f"Length: {duration_desc}. Use beat cues: (CUT), (VO), (MUSIC UP), (DROP), (TEXT ON SCREEN). "
                        f"Hook in first 2 seconds. No greetings, no CTA, no headings. End with a human question or punchline."
                    )
                else:
                    duration_desc = f"{effective_length_minutes} minutes"
                    simple_prompt = f"""Create a COMPLETE {duration_desc} YouTube script about {topic} in {lang_desc}. 
                    Make it informative and engaging for the target audience. 
                    Include: Hook, introduction, main content, and a conversational CTA.
                    IMPORTANT: Generate the ENTIRE script from start to finish. Do not truncate or cut off mid-sentence."""
                
                response = self.model.generate_content(
                    simple_prompt,
                    generation_config=call_generation_config
                )
                
                if not response.candidates or not response.text:
                    return {
                        'success': False,
                        'error': 'Content blocked by safety filters. Please try a different topic or rephrase your request.',
                        'metadata': {
                            'topic': topic, 
                            'length_minutes': effective_length_minutes if content_format != "short-form" else None,
                            'length_seconds': effective_length_seconds if content_format == "short-form" else None
                        }
                    }
            
            # Check safety ratings safely
            if getattr(response, 'candidates', None):
                candidate = response.candidates[0]
                if getattr(candidate, 'safety_ratings', None):
                    blocked_categories = []
                    for rating in candidate.safety_ratings:
                        if getattr(rating, 'probability', None) in ['HIGH', 'MEDIUM']:
                            blocked_categories.append(f"{rating.category}: {rating.probability}")
                    if blocked_categories:
                        return {
                            'success': False,
                            'error': f'Content blocked by safety filters: {", ".join(blocked_categories)}. Try rephrasing your topic or context.',
                            'metadata': {'topic': topic, 'length_minutes': length_minutes}
                        }

            # Extract text safely (if not already extracted above)
            if not script_text or len(script_text.strip()) == 0:
                script_text = self._extract_text_from_response(response)
                if not script_text:
                    # Check if it was due to MAX_TOKENS (fallback check)
                    if getattr(response, 'candidates', None) and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        finish_reason_check = getattr(candidate, 'finish_reason', None)
                        if finish_reason_check == "MAX_TOKENS":
                            return {
                                'success': False,
                                'error': 'No text content generated due to token limit. The prompt may be too large. Try reducing the outline, facts, or topic complexity.',
                                'metadata': {
                                    'topic': topic,
                                    'length_minutes': effective_length_minutes if content_format != "short-form" else None,
                                    'length_seconds': effective_length_seconds if content_format == "short-form" else None,
                                    'finish_reason': 'MAX_TOKENS'
                                }
                            }
                    
                    return {
                        'success': False,
                        'error': 'No text content generated. The response may have been blocked or filtered.',
                        'metadata': {'topic': topic, 'length_minutes': length_minutes}
                    }
            
            # Continuation if early stop or short vs cap
            accumulated_text = script_text
            needs_more = len(accumulated_text.split()) < int(hard_word_cap * 0.85)
            expected_items = self._extract_expected_item_count(topic)
            if expected_items:
                current_items = self._count_list_items(accumulated_text)
                if current_items < expected_items:
                    needs_more = True
            if needs_more:
                # first generic continuation
                accumulated_text = self._attempt_continuation(accumulated_text, hard_word_cap, call_generation_config)
                # guided continuation for remaining items
                if expected_items:
                    current_items = self._count_list_items(accumulated_text)
                    if current_items < expected_items and (hard_word_cap - len(accumulated_text.split())) > 60:
                        guidance = f"Continue with items {current_items+1} to {expected_items}. Keep each item concise and balanced. Do not repeat. "
                        accumulated_text = self._attempt_guided_continuation(accumulated_text, hard_word_cap, call_generation_config, guidance)

            # Clean and post-process the generated script
            cleaned_text = self._clean_response_text(accumulated_text)
            processed_script = self._post_process_script(
                cleaned_text, 
                effective_length_minutes, 
                effective_length_seconds if content_format == "short-form" else None,
                hard_word_cap
            )
            
            # Generate structured output
            structured_script = self._parse_script_to_structure(
                processed_script['script'],
                processed_script['timing_suggestions'],
                effective_length_minutes,
                effective_length_seconds if content_format == "short-form" else None
            )
            
            return {
                    'success': True,
                    'script': processed_script['script'],  # Plain text for backward compatibility
                    'structured': structured_script,  # Structured JSON format
                    'metadata': {
                        'topic': topic,
                        'length_minutes': effective_length_minutes,
                        'length_seconds': effective_length_seconds if content_format == "short-form" else None,
                        'content_format': content_format,
                        'estimated_word_count': processed_script['word_count'],
                        'estimated_speaking_time': processed_script['speaking_time'],
                        'tone_used': tone,
                        'target_audience': target_audience,
                        'content_type': content_type,
                        'creator_style_used': creator_style,
                        'generation_time_seconds': round(generation_time, 2)
                    },
                    'timing_suggestions': processed_script['timing_suggestions'],
                    'creator_patterns_applied': processed_script['pattern_markers']
                }
                
        except Exception as e:
            print(f"Script generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'topic': topic, 
                    'length_minutes': effective_length_minutes if effective_length_minutes else None,
                    'length_seconds': effective_length_seconds if content_format == "short-form" else None
                }
            }
    
    def _build_generation_prompt(self, 
                                topic: str, 
                                length_minutes: float, 
                                length_seconds: float,
                                tone: str, 
                                target_audience: str,
                                content_type: str,
                                content_format: str,
                                style_mode: Optional[str],
                                creator_style: Optional[str],
                                outline: Optional[str],
                                facts: Optional[str],
                                hard_word_cap: Optional[int] = None,
                                language_mix: str = "Balanced",
                                force_hinglish_ascii: bool = True) -> str:
        """Build comprehensive prompt for script generation"""
        
        # Base prompt structure
        prompt_parts = []
        
        # Determine cinematic short-form behavior
        is_cinematic = (content_format == "short-form") or (style_mode == "cinematic_reel")
        
        # System context - adjust based on mode and language mix
        if is_cinematic:
            base_ctx = "You are an expert short-form scriptwriter for cinematic reels. Prioritize authenticity, rhythm, and emotional impact over explanations."
            if language_mix == "English":
                prompt_parts.append(base_ctx + " Write in English.")
            else:
                prompt_parts.append(base_ctx + " Write in Hinglish (Hindi + English mix).")
        else:
            if language_mix == "English":
                prompt_parts.append("""You are an expert YouTube script writer specializing in creating authentic English content. Create educational, informative, and family-friendly content.""")
            else:
                prompt_parts.append("""You are an expert YouTube script writer specializing in creating authentic Hinglish (Hindi + English mix) content for Indian channels. Create educational, informative, and family-friendly content.""")
        
        # Content format context
        if content_format == "short-form":
            # Cinematic reel defaults for short-form
            prompt_parts.append(f"""CONTENT FORMAT: Short-form reel ({length_seconds} seconds).
- Cinematic, fast-cut visuals with voiceover (VO)
- No greetings/CTAs; no headings or lists
- Use beat cues: (CUT), (VO), (SFX), (MUSIC UP), (DROP), (TEXT ON SCREEN)
- Hook in 1–2s; end on a question/punchline
""")
        else:
            prompt_parts.append(f"""CONTENT FORMAT: Long-form video ({length_minutes} minutes).
- Structured content with clear sections
- Detailed explanations and examples
- Engaging throughout the duration
""")
        
        # RAG: Retrieve similar scripts for context injection
        rag_context = ""
        if self.rag_system and self.rag_system._initialized:
            try:
                # Build query from topic and content type
                query = f"{topic} {content_type}"
                if creator_style:
                    query += f" {creator_style}"
                
                # Retrieve similar scripts
                similar_scripts = self.rag_system.retrieve_similar(query, n_results=3)
                
                if similar_scripts:
                    rag_context = "\n\nREFERENCE SCRIPTS (for style and structure guidance):\n"
                    rag_context += "Study these similar successful scripts to understand the style, structure, and language patterns:\n\n"
                    
                    for idx, script_data in enumerate(similar_scripts, 1):
                        metadata = script_data.get('metadata', {})
                        doc_text = script_data.get('document', '')[:800]  # Limit to 800 chars per script
                        rag_context += f"--- Reference Script {idx} ---\n"
                        rag_context += f"Title: {metadata.get('title', 'Unknown')}\n"
                        rag_context += f"Creator: {metadata.get('uploader', 'Unknown')}\n"
                        rag_context += f"Excerpt: {doc_text}\n\n"
                    
                    rag_context += "IMPORTANT: Use these as style references. Match the tone, language patterns, and structure, but create original content for the new topic.\n"
            except Exception as e:
                print(f"[WARN] RAG retrieval failed: {e}")
        
        # Rich creator-specific context with examples
        if creator_style and creator_style in self.creator_styles:
            creator_data = self.creator_styles[creator_style]
            creator_context = self._build_creator_specific_context(
                creator_data, 
                creator_style, 
                topic, 
                content_type,
                language_mix,
                content_format
            )
            prompt_parts.append(creator_context)
        elif not creator_style:
            # Auto-Select mode: use aggregated context
            aggregated_context = self._build_aggregated_creator_context(topic, content_type, language_mix)
            prompt_parts.append(aggregated_context)
        
        # Add RAG context after creator context
        if rag_context:
            prompt_parts.append(rag_context)
        
        # Main instruction
        if content_format == "short-form":
            duration_desc = f"{length_seconds} seconds (approximately {hard_word_cap} words)"
        else:
            duration_desc = f"{length_minutes} minutes (approximately {hard_word_cap} words)"
        
        prompt_parts.append(f"""
SCRIPT REQUIREMENTS:
- Topic: {topic}
- Duration: {duration_desc}
- Tone: {tone}
- Target Audience: {target_audience}
- Content Type: {content_type}
- Language: {language_mix}

IMPORTANT HARD LIMIT: Write no more than {hard_word_cap} words. Stop before exceeding this word cap. The script must be complete but concise.
IMPORTANT: Generate a COMPLETE script from start to finish, but keep it within the word cap. Do not cut off mid-sentence.
""")
        if not is_cinematic:
            prompt_parts.append("Include all sections: Hook, Introduction, Main Content, and Conclusion with Call-to-Action.")
        else:
            prompt_parts.append("Write as continuous VO with beat cues. No greetings/CTAs/headings. End with a human question or punchline.")
        
        # Outline enhancement
        if outline:
            prompt_parts.append(f"""
OUTLINE ENHANCEMENT:
The user provided this outline:
{outline}

IMPORTANT: Improve and enhance this outline while keeping the same flow and structure. Make it more engaging, detailed, and compelling. Use this as the foundation for the script but expand upon it naturally.
""")
        
        # Facts and information
        if facts:
            prompt_parts.append(f"""
FACTS & INFORMATION:
Use these facts as core information around which to build the script:
{facts}

Ensure all these facts are naturally integrated into the script. Use them as the foundation for accurate and informative content.
""")
        
        # Style guidelines
        tone_description = self._get_tone_guidelines(tone)
        content_guidelines = self._get_content_type_guidelines(content_type)
        
        prompt_parts.append(f"""
STYLE GUIDELINES:
{tone_description}

{content_guidelines}

SCRIPT STRUCTURE:
1. Hook (0-15 seconds): Engaging opening question or statement
2. Intro (15-30 seconds): Channel greeting and video preview
3. Main Content ({length_minutes - 1}-{length_minutes - 0.5} minutes): Core topic discussion
4. Call-to-Action (15-30 seconds): Like, subscribe, notification bell
5. Outro (15-30 seconds): Channel promotion and preview

HINGLISH LANGUAGE PATTERNS:
- Natural mix of Hindi and English
- Common phrases: "दोस्तों", "भाई", "तो यहाँ पर", "सुनिए"
- Technical terms in English, explanations in Hindi
- Engaging expressions and enthusiasm markers
""")
        
        # Language-specific instructions with strong enforcement
        if language_mix == "English":
            prompt_parts.append("\nIMPORTANT: Write the entire script in English only. Do not use Hindi or Hinglish. Use clear, engaging English throughout.")
        elif language_mix == "Hindi Heavy":
            prompt_parts.append("""
CRITICAL LANGUAGE REQUIREMENT - HINDI HEAVY MODE:
- 70-80% of words MUST be Hindi/Hinglish (Hindi written in Latin script like 'aap', 'kaise', 'bahut')
- Most sentences should START in Hindi
- Use Hindi grammar structure: "Aapko yeh dekhna chahiye" NOT "You should see this"
- English ONLY for: technical terms (iPhone, Android, YouTube), brand names, or when no Hindi equivalent exists
- Use Hindi for: verbs (dekhna, karna, banna), adjectives (accha, bahut, kaafi), conjunctions (aur, par, lekin), common nouns

CORRECT EXAMPLES (70-80% Hindi):
- "Dosto, aaj main aapko bataunga ki yeh phone kaise hai. Iska camera bahut accha hai aur performance bhi kamaal ka hai."
- "Aapko yeh movie dekhni chahiye kyunki iska story bahut interesting hai aur acting bhi excellent hai."

WRONG EXAMPLES (too much English):
- "Hey guys, today I will tell you about this phone. Its camera is good and performance is also great."
- "You should watch this movie because its story is interesting and acting is excellent."

WORD COUNT CHECK: After writing, mentally count - at least 70% of words should be Hindi/Hinglish words.
""")
            if force_hinglish_ascii:
                prompt_parts.append("\nIMPORTANT: Write Hindi words in Latin script only (no Devanagari). Example: 'aap kya kar rahe ho', 'dosto', 'dekhiye'.")
        elif language_mix == "Balanced":
            prompt_parts.append("""
CRITICAL LANGUAGE REQUIREMENT - BALANCED MODE:
- Approximately 50% Hindi/Hinglish and 50% English words
- Mix naturally: Start sentences in either language, alternate between them organically
- Use Hindi for: conversational parts (greetings, expressions), emotions, common phrases
- Use English for: technical terms, specific concepts, modern terms, precise descriptions
- Pattern: Natural code-switching like real Hinglish speakers do in daily conversation

CORRECT EXAMPLES (50-50 balance):
- "Dosto, today we're going to review this amazing phone. Iska camera quality bahut accha hai, and the processor bhi kaafi powerful hai."
- "Hey guys, aaj hum baat karenge top 5 movies ke baare mein. These are some amazing films jo aapko definitely dekhni chahiye."

WRONG EXAMPLES (too unbalanced):
- "Dosto, aaj main aapko bataunga ki yeh phone kaise hai. Iska camera accha hai aur performance bhi great hai." (too much Hindi)
- "Hey guys, today I'll tell you about this phone. Its camera is good and performance is also great." (too much English)

WORD COUNT CHECK: Aim for roughly equal Hindi and English word count throughout the script.
""")
            if force_hinglish_ascii:
                prompt_parts.append("\nIMPORTANT: Write Hindi words in Latin script only (no Devanagari). Example: 'aap', 'kaise', 'bahut'.")
            # Naturalness guard to avoid robotic enforcement
            prompt_parts.append("\nPRIORITY: Sound like a real human speaking. If strict 50–50 enforcement conflicts with natural voice, prefer flow and stay near the target mix overall.")
        else:
            # Default/Other - generic Hinglish
            if force_hinglish_ascii:
                prompt_parts.append("\nIMPORTANT: Write in Hinglish using Latin letters only (no Devanagari). Example: 'aap kya kar rahe ho', 'dosto', 'performance'. Keep it natural.")
            else:
                prompt_parts.append("\nIMPORTANT: Write in Hinglish (Hindi + English mix). Use natural language mixing.")
        
        prompt_parts.append("\nGenerate an engaging, authentic YouTube script that sounds natural and conversational.")
        if not is_cinematic:
            prompt_parts.append("\nSCRIPT STRUCTURE FORMATTING:")
            prompt_parts.append("Use section markers to clearly separate parts of your script. Format as:")
            prompt_parts.append("[HOOK]")
            prompt_parts.append("Your hook content here...")
            prompt_parts.append("[INTRO]")
            prompt_parts.append("Your intro content here...")
            prompt_parts.append("[MAIN CONTENT]")
            prompt_parts.append("Your main content here...")
            prompt_parts.append("[CTA]")
            prompt_parts.append("Your call-to-action here...")
            prompt_parts.append("[OUTRO]")
            prompt_parts.append("Your outro content here...")
        else:
            # Beat-timed cinematic guidance for short-form
            prompt_parts.append("\nWRITE AS CONTINUOUS VOICEOVER WITH BEAT CUES:")
            prompt_parts.append("- (CUT 0.0s) Hook line that compels attention")
            prompt_parts.append("- (MUSIC UP) escalate tension")
            prompt_parts.append("- Use 3–5 quick beats to cover key contrasts or highlights")
            prompt_parts.append("- (DROP) deliver the emotional peak or twist")
            prompt_parts.append("- (TEXT ON SCREEN) 4–6 words max")
            prompt_parts.append("- End with an open-ended question or punchline; no formal CTA")
        prompt_parts.append("\nCRITICAL: Make sure to complete the entire script. Do not stop mid-sentence or leave sections incomplete. The script must be complete from beginning to end.")
        prompt_parts.append("\nCOMPLETION REQUIREMENTS:\n- If the topic implies a list (e.g., \"Top 5 laptops\"), include exactly that many fully detailed items with consistent headings and balanced detail per item.\n- If you run out of room, compress wording rather than dropping items.\n- Ensure the script ends with a clear outro/CTA and a complete final sentence.")
        
        return "\n".join(prompt_parts)

    def _extract_text_from_response(self, response) -> Optional[str]:
        """Safely extract text from a Gemini response without triggering quick-accessor errors."""
        # Try quick accessor
        try:
            if getattr(response, 'text', None):
                return response.text
        except Exception:
            pass
        # Try candidates and parts
        try:
            if getattr(response, 'candidates', None):
                for cand in response.candidates:
                    content = getattr(cand, 'content', None)
                    if content and getattr(content, 'parts', None):
                        # Concatenate all text parts if available
                        parts_text = []
                        for p in content.parts:
                            t = getattr(p, 'text', None)
                            if t:
                                parts_text.append(t)
                        if parts_text:
                            return "\n".join(parts_text)
        except Exception:
            pass
        # Fallback to string
        try:
            s = str(response)
            return s if s else None
        except Exception:
            return None

    def _attempt_continuation(self, current_text: str, hard_word_cap: int, generation_config: Dict) -> str:
        """If the model stopped early, request continuation until cap/outro reached."""
        try:
            remaining_words = max(0, hard_word_cap - len(current_text.split()))
            if remaining_words < 50:
                return current_text
            tail = " ".join(current_text.split()[-80:])
            continuation_prompt = (
                f"Continue the script from where it stopped. Do not repeat any sentences. "
                f"Finish the remaining sections and end with a proper outro. "
                f"You have approximately {remaining_words} words remaining (total cap {hard_word_cap}). "
                f"Here are the last lines to continue from:\n" + tail
            )
            more = self.model.generate_content(continuation_prompt, generation_config=generation_config)
            if more and getattr(more, 'text', None):
                combined = current_text.rstrip() + "\n\n" + more.text.strip()
                return self._truncate_to_words_sentence_aware(combined, hard_word_cap)
        except Exception:
            return current_text
        return current_text

    def _attempt_guided_continuation(self, current_text: str, hard_word_cap: int, generation_config: Dict, guidance: str) -> str:
        """Continuation with guidance for remaining list items."""
        try:
            remaining_words = max(0, hard_word_cap - len(current_text.split()))
            if remaining_words < 50:
                return current_text
            tail = " ".join(current_text.split()[-80:])
            continuation_prompt = (
                guidance +
                f"You have approximately {remaining_words} words remaining (total cap {hard_word_cap}). "
                f"Here are the last lines to continue from:\n" + tail
            )
            more = self.model.generate_content(continuation_prompt, generation_config=generation_config)
            if more and getattr(more, 'text', None):
                combined = current_text.rstrip() + "\n\n" + more.text.strip()
                return self._truncate_to_words_sentence_aware(combined, hard_word_cap)
        except Exception:
            return current_text
        return current_text

    def _extract_expected_item_count(self, topic: str, additional_context: Optional[str] = None) -> Optional[int]:
        """Extract expected item count from topic/context (e.g., 'Top 5', '5 laptops')."""
        import re
        text = topic if topic else ""
        patterns = [r"top\s+(\d+)", r"\b(\d+)\s*(?:items?|laptops?|phones?|mobiles?|tips?|points?)\b"]
        candidates = []
        for pat in patterns:
            for m in re.findall(pat, text, flags=re.IGNORECASE):
                try:
                    candidates.append(int(m))
                except Exception:
                    pass
        return max(candidates) if candidates else None

    def _count_list_items(self, text: str) -> int:
        """Heuristic count of enumerated items in the current script."""
        import re
        lines = [ln.strip() for ln in text.split('\n')]
        count = 0
        for ln in lines:
            if re.match(r"^(?:\d+[\).]|[-*]\s)\s*", ln):
                count += 1
            elif re.match(r"^\*\*\s*\w+\s*\d+\s*\*\*", ln):
                count += 1
            elif re.match(r"^(?:laptop|phone|item)\s*\d+[:\-]", ln, flags=re.IGNORECASE):
                count += 1
        return count
    
    def _clean_response_text(self, text: str) -> str:
        """Clean and fix encoding issues in the response text"""
        import re
        
        # First, try to decode and re-encode to fix encoding issues
        try:
            # Remove any replacement characters and fix common encoding issues
            cleaned_text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            cleaned_text = text
        
        # Remove problematic Unicode characters that cause display issues
        cleaned_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', cleaned_text)  # Remove zero-width characters
        # IMPORTANT: Do NOT collapse spaces between Devanagari characters; that destroys word boundaries.
        
        # Remove any remaining replacement characters
        cleaned_text = re.sub(r'\uFFFD+', '', cleaned_text)
        
        # Clean up excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)  # Remove excessive line breaks
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # Normalize spaces
        
        # Ensure proper line breaks
        cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')
        
        return cleaned_text
    
    def _get_language_mix_description(self, creator_data: Dict) -> str:
        """Get language mix description for creator"""
        pref = creator_data.get('language_preferences', {})
        
        if pref.get('hindi_dominant', 0) > pref.get('english_dominant', 0):
            return "Hindi-dominant Hinglish (more Hindi expressions, Hindi grammatical structure)"
        elif pref.get('english_dominant', 0) > pref.get('hindi_dominant', 0):
            return "English-dominant Hinglish (more English words, English grammatical structure)"
        else:
            return "Balanced Hinglish (equal mix of Hindi and English)"
    
    def _get_tone_description(self, creator_data: Dict) -> str:
        """Get tone description for creator"""
        tone_profile = creator_data.get('tone_profile', {})
        
        parts = []
        if tone_profile.get('enthusiasm', 0) > 7:
            parts.append("Highly enthusiastic")
        if tone_profile.get('technical_depth', 0) > 7:
            parts.append("Technical and detailed")
        if tone_profile.get('friendliness', 0) > 7:
            parts.append("Friendly and approachable")
        
        return ", ".join(parts) if parts else "Balanced conversational tone"
    
    def _select_relevant_examples(self, creator_data: Dict, topic: str, content_type: str = None, language_mix: str = "Balanced") -> List[Dict]:
        """Select best matching transcript examples for the given topic and content type.
        Enhanced with RAG-retrieved best-performing scripts for few-shot learning.
        """
        import re
        
        examples = []
        
        # First, try to get best-performing scripts from RAG
        if self.rag_system and self.rag_system._initialized:
            try:
                # Get best performing scripts (by view count)
                best_scripts = self.rag_system.get_best_performing_scripts(n=3)
                
                # Also get similar scripts by topic
                similar_scripts = self.rag_system.retrieve_similar(
                    query=f"{topic} {content_type or ''}",
                    n_results=2
                )
                
                # Combine and convert to example format
                for script_data in best_scripts + similar_scripts:
                    doc_text = script_data.get('document', '')
                    metadata = script_data.get('metadata', {})
                    
                    # Extract content from document
                    if 'Content:' in doc_text:
                        content_part = doc_text.split('Content:')[1].strip()
                        if len(content_part) > 200:  # Only meaningful examples
                            examples.append({
                                'text': content_part[:800],  # Limit length
                                'title': metadata.get('title', 'High-performing video'),
                                'source': 'rag_best_performing',
                                'view_count': metadata.get('view_count', 0)
                            })
            except Exception as e:
                print(f"[WARN] RAG few-shot retrieval failed: {e}")
        
        # Fallback to creator's transcript examples if RAG didn't provide enough
        if len(examples) < 2 and creator_data.get('transcript_examples'):
            examples.extend(creator_data['transcript_examples'])
        
        if not examples:
            return []
        
        # Get creator's language preferences for scoring
        lang_prefs = creator_data.get('language_preferences', {})
        hindi_dominant_count = lang_prefs.get('hindi_dominant', 0)
        balanced_count = lang_prefs.get('balanced', 0)
        english_dominant_count = lang_prefs.get('english_dominant', 0)
        total_videos = hindi_dominant_count + balanced_count + english_dominant_count
        
        # Calculate creator's average Hindi ratio
        if total_videos > 0:
            creator_hindi_ratio = (hindi_dominant_count + balanced_count * 0.5) / total_videos
        else:
            creator_hindi_ratio = 0.5  # Default to balanced
        
        # Extract keywords from topic
        topic_lower = topic.lower()
        topic_words = set(re.findall(r'\b\w+\b', topic_lower))
        
        # Score examples based on relevance
        scored_examples = []
        for example in examples:
            score = 0
            example_text_lower = example['text'].lower()
            example_title_lower = example.get('title', '').lower()
            
            # Check for topic word matches in text
            for word in topic_words:
                if len(word) > 3:  # Only meaningful words
                    if word in example_text_lower:
                        score += 2
                    if word in example_title_lower:
                        score += 3  # Title matches are more important

            # Emotional/impact cues boost (useful for short-form reels)
            emotional_cues = [
                'wow', 'shocking', 'incredible', 'unbelievable', 'controversy', 'dramatic',
                'guts', 'passion', 'heart', 'inspiring', 'legend', 'brutal', 'twist', 'reveal'
            ]
            exclamations = example_text_lower.count('!')
            cue_hits = sum(1 for w in emotional_cues if w in example_text_lower)
            if exclamations > 0 or cue_hits > 0:
                score += min(3, exclamations + cue_hits)
            
            # Content type matching
            if content_type:
                content_keywords = {
                    'review': ['review', 'रीव्यू', 'test', 'टेस्ट', 'spec', 'feature'],
                    'comparison': ['vs', 'versus', 'कंपेयर', 'compare', 'difference'],
                    'unboxing': ['unboxing', 'अनबॉक्सिंग', 'first look', 'unbox'],
                    'tutorial': ['guide', 'गाइड', 'tips', 'how to', 'tutorial'],
                    'general': []
                }
                keywords_for_type = content_keywords.get(content_type, [])
                for keyword in keywords_for_type:
                    if keyword in example_text_lower or keyword in example_title_lower:
                        score += 1
            
            # Language mix preference scoring
            if language_mix == "Hindi Heavy":
                # Boost examples from Hindi-dominant creators
                if creator_hindi_ratio > 0.65:
                    score += 5  # Strong boost for Hindi-dominant creators
                elif creator_hindi_ratio > 0.5:
                    score += 2  # Moderate boost for balanced creators
                # Also check example text for Hindi words
                hindi_words_in_example = len(re.findall(r'\b(aap|kaise|bahut|accha|dekhiye|suniye|dosto|bhai|yaar|kyunki|isliye|iska|uska|hoga|hoga|hai|hain)\b', example_text_lower))
                if len(example_text_lower.split()) > 0:
                    hindi_ratio_in_example = hindi_words_in_example / len(example_text_lower.split())
                    if hindi_ratio_in_example > 0.6:
                        score += 3
            elif language_mix == "Balanced":
                # Prefer examples from creators with balanced mix
                if 0.4 < creator_hindi_ratio < 0.6:
                    score += 3  # Boost for balanced creators
            
            # Prefer start examples (they show intro style better)
            if example.get('type') == 'start':
                score += 1
            
            scored_examples.append((score, example))
        
        # Sort by score (highest first) and return top 2-3 examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Select 2-3 best examples, but ensure diversity
        selected = []
        selected_titles = set()
        
        for score, example in scored_examples:
            if len(selected) >= 3:
                break
            title_key = example.get('title', '')[:30]  # Use title prefix to avoid duplicates
            if title_key not in selected_titles or score > 3:  # Allow if high score
                selected.append(example)
                selected_titles.add(title_key)
        
        # If we don't have enough, fill with highest scoring remaining
        if len(selected) < 2:
            for score, example in scored_examples:
                if len(selected) >= 3:
                    break
                if example not in selected:
                    selected.append(example)
        
        # Always return at least 2 examples if available
        return selected[:3] if len(selected) >= 2 else selected
    
    def _build_creator_specific_context(self, creator_data: Dict, creator_name: str, topic: str, content_type: str = None, language_mix: str = "Balanced", content_format: str = "long-form") -> str:
        """Build rich creator-specific context with examples, patterns, and phrases for prompt"""
        context_parts = []
        
        # Select relevant examples with language mix preference
        selected_examples = self._select_relevant_examples(creator_data, topic, content_type, language_mix)
        
        # Language mix description
        lang_mix_desc = self._get_language_mix_description(creator_data)
        tone_desc = self._get_tone_description(creator_data)
        
        # Build context header
        context_parts.append(f"""
=== CREATOR STYLE CONTEXT ===
You are generating a YouTube script in the authentic style of {creator_name}.

CREATOR CHARACTERISTICS:
- Language Style: {lang_mix_desc}
- Tone Profile: {tone_desc}
- Speaking Pace: ~{creator_data.get('avg_speaking_pace', 140):.0f} words per minute
""")
        
        # Add transcript examples (real examples from creator)
        if selected_examples:
            context_parts.append(f"AUTHENTIC EXAMPLE OUTPUTS (from actual {creator_name} videos):\n")
            # Trim to 2 examples, 500 chars each for tighter prompts
            for i, example in enumerate(selected_examples[:2], 1):
                example_text = example['text'][:500]
                example_title = example.get('title', 'Video example')
                context_parts.append(f"EXAMPLE {i} (from: {example_title}):")
                context_parts.append(f"{example_text}...")
                context_parts.append("")
            
            context_parts.append("IMPORTANT: Study these examples carefully. Notice:")
            context_parts.append("- The exact speaking style and flow")
            context_parts.append("- How sentences are structured")
            context_parts.append("- The natural mix of Hindi and English")
            context_parts.append("- Specific phrases and expressions used")
            context_parts.append("")
        
        # Add intro patterns (skip for short-form)
        intro_patterns = creator_data.get('intro_patterns', [])
        if intro_patterns and content_format != "short-form":
            context_parts.append(f"AUTHENTIC INTRO PATTERNS (how {creator_name} starts videos):")
            # Show top 5 intro patterns
            for i, intro in enumerate(intro_patterns[:5], 1):
                # Limit each intro to 150 chars
                intro_snippet = intro[:150] + "..." if len(intro) > 150 else intro
                context_parts.append(f"  {i}. {intro_snippet}")
            context_parts.append("")
        
        # Add outro patterns (skip for short-form)
        outro_patterns = creator_data.get('outro_patterns', [])
        if outro_patterns and content_format != "short-form":
            context_parts.append(f"AUTHENTIC OUTRO PATTERNS (how {creator_name} ends videos):")
            # Show top 5 outro patterns
            for i, outro in enumerate(outro_patterns[:5], 1):
                outro_snippet = outro[:150] + "..." if len(outro) > 150 else outro
                context_parts.append(f"  {i}. {outro_snippet}")
            context_parts.append("")
        
        # Add common expressions and unique phrases
        common_expressions = creator_data.get('common_expressions', {})
        transition_phrases = creator_data.get('transition_phrases', {})
        
        if common_expressions or transition_phrases:
            context_parts.append(f"STYLE MARKERS - Common expressions and phrases {creator_name} uses:")
            
            # Top common expressions
            if common_expressions:
                top_expressions = list(common_expressions.keys())[:7]
                context_parts.append(f"  Common Expressions: {', '.join(top_expressions)}")
            
            # Top transition phrases
            if transition_phrases:
                top_transitions = list(transition_phrases.keys())[:5]
                context_parts.append(f"  Transition Phrases: {', '.join(top_transitions)}")
            
            context_parts.append("")
            context_parts.append(f"IMPORTANT: Use these expressions naturally throughout the script, just like {creator_name} does.")
            context_parts.append("")
        
        # Add content style markers
        content_style = creator_data.get('content_style', [])
        if content_style:
            context_parts.append(f"Content Style Focus: {', '.join(list(content_style)[:5])}")
            context_parts.append("")
        
        context_parts.append("CRITICAL INSTRUCTIONS:")
        context_parts.append(f"1. Write EXACTLY like {creator_name} would write/speak")
        context_parts.append("2. Use the same expressions, phrases, and speaking patterns from the examples above")
        context_parts.append("3. Match the speaking pace and flow style")
        context_parts.append("4. Replicate the intro/outro patterns naturally")
        context_parts.append("5. Make it sound authentic - as if {creator_name} actually created this script")
        context_parts.append("")
        context_parts.append("=== END CREATOR STYLE CONTEXT ===\n")
        
        return "\n".join(context_parts)
    
    def _select_diverse_examples_from_all(self, topic: str, content_type: str = None, language_mix: str = "Balanced", max_creators: int = 4) -> List[Dict]:
        """Select diverse examples from multiple creators that match the topic"""
        import re
        
        if not self.creator_styles:
            return []
        
        # Score examples from all creators
        all_scored_examples = []
        
        for creator_name, creator_data in self.creator_styles.items():
            examples = creator_data.get('transcript_examples', [])
            if not examples:
                continue
            
            # Get creator's language preferences
            lang_prefs = creator_data.get('language_preferences', {})
            hindi_dominant_count = lang_prefs.get('hindi_dominant', 0)
            balanced_count = lang_prefs.get('balanced', 0)
            english_dominant_count = lang_prefs.get('english_dominant', 0)
            total_videos = hindi_dominant_count + balanced_count + english_dominant_count
            
            if total_videos > 0:
                creator_hindi_ratio = (hindi_dominant_count + balanced_count * 0.5) / total_videos
            else:
                creator_hindi_ratio = 0.5
            
            # Extract keywords from topic
            topic_lower = topic.lower()
            topic_words = set(re.findall(r'\b\w+\b', topic_lower))
            
            # Score each example
            for example in examples:
                score = 0
                example_text_lower = example['text'].lower()
                example_title_lower = example.get('title', '').lower()
                
                # Topic relevance scoring
                for word in topic_words:
                    if len(word) > 3:
                        if word in example_text_lower:
                            score += 2
                        if word in example_title_lower:
                            score += 3
                
                # Content type matching
                if content_type:
                    content_keywords = {
                        'review': ['review', 'रीव्यू', 'test', 'टेस्ट', 'spec', 'feature'],
                        'comparison': ['vs', 'versus', 'कंपेयर', 'compare', 'difference'],
                        'unboxing': ['unboxing', 'अनबॉक्सिंग', 'first look', 'unbox'],
                        'tutorial': ['guide', 'गाइड', 'tips', 'how to', 'tutorial'],
                        'general': []
                    }
                    keywords_for_type = content_keywords.get(content_type, [])
                    for keyword in keywords_for_type:
                        if keyword in example_text_lower or keyword in example_title_lower:
                            score += 1
                
                # Language mix preference scoring
                if language_mix == "Hindi Heavy":
                    if creator_hindi_ratio > 0.65:
                        score += 5
                    elif creator_hindi_ratio > 0.5:
                        score += 2
                    # Check example text for Hindi words
                    hindi_words_in_example = len(re.findall(r'\b(aap|kaise|bahut|accha|dekhiye|suniye|dosto|bhai|yaar|kyunki|isliye|iska|uska|hoga|hai|hain)\b', example_text_lower))
                    if len(example_text_lower.split()) > 0:
                        hindi_ratio_in_example = hindi_words_in_example / len(example_text_lower.split())
                        if hindi_ratio_in_example > 0.6:
                            score += 3
                elif language_mix == "Balanced":
                    if 0.4 < creator_hindi_ratio < 0.6:
                        score += 3
                
                # Prefer start examples
                if example.get('type') == 'start':
                    score += 1
                
                all_scored_examples.append({
                    'score': score,
                    'example': example,
                    'creator': creator_name
                })
        
        # Sort by score (highest first)
        all_scored_examples.sort(key=lambda x: x['score'], reverse=True)
        
        # Select diverse examples (max 1-2 per creator, total 3-4 examples)
        selected = []
        creators_used = {}
        
        for item in all_scored_examples:
            if len(selected) >= 4:  # Max 4 examples total
                break
            
            creator = item['creator']
            example = item['example']
            
            # Limit to 2 examples per creator
            if creator not in creators_used:
                creators_used[creator] = 0
            
            if creators_used[creator] < 2 and item['score'] > 0:  # Only if relevant
                # Avoid duplicate titles
                title_key = example.get('title', '')[:30]
                existing_titles = [e.get('title', '')[:30] for e in selected]
                
                if title_key not in existing_titles:
                    selected.append({
                        'example': example,
                        'creator': creator
                    })
                    creators_used[creator] += 1
        
        # If we don't have enough, fill with highest scoring remaining
        if len(selected) < 3:
            for item in all_scored_examples:
                if len(selected) >= 4:
                    break
                
                creator = item['creator']
                example = item['example']
                title_key = example.get('title', '')[:30]
                existing_titles = [e.get('example', {}).get('title', '')[:30] for e in selected]
                
                if title_key not in existing_titles:
                    if creators_used.get(creator, 0) < 2:
                        selected.append({
                            'example': example,
                            'creator': creator
                        })
                        creators_used[creator] = creators_used.get(creator, 0) + 1
        
        return selected[:4]  # Return max 4 diverse examples
    
    def _aggregate_creator_stats(self) -> Dict:
        """Aggregate statistics from all creators for blended style"""
        if not self.creator_styles:
            return {}
        
        aggregated = {
            'avg_speaking_pace': 0,
            'common_expressions': {},
            'transition_phrases': {},
            'intro_patterns': [],
            'outro_patterns': [],
            'tone_profile': {'enthusiasm': 0, 'technical_depth': 0, 'friendliness': 0},
            'language_preferences': {'hindi_dominant': 0, 'english_dominant': 0, 'balanced': 0}
        }
        
        from collections import Counter
        expression_counter = Counter()
        transition_counter = Counter()
        
        total_creators = len(self.creator_styles)
        
        for creator_name, creator_data in self.creator_styles.items():
            # Aggregate speaking pace
            aggregated['avg_speaking_pace'] += creator_data.get('avg_speaking_pace', 140)
            
            # Aggregate expressions
            expressions = creator_data.get('common_expressions', {})
            for expr, count in expressions.items():
                expression_counter[expr] += count
            
            # Aggregate transition phrases
            transitions = creator_data.get('transition_phrases', {})
            for trans, count in transitions.items():
                transition_counter[trans] += count
            
            # Collect intro/outro patterns (top 3 from each)
            intro_patterns = creator_data.get('intro_patterns', [])[:3]
            aggregated['intro_patterns'].extend(intro_patterns)
            
            outro_patterns = creator_data.get('outro_patterns', [])[:3]
            aggregated['outro_patterns'].extend(outro_patterns)
            
            # Aggregate tone profile
            tone_profile = creator_data.get('tone_profile', {})
            for tone in ['enthusiasm', 'technical_depth', 'friendliness']:
                aggregated['tone_profile'][tone] += tone_profile.get(tone, 0)
            
            # Aggregate language preferences
            lang_prefs = creator_data.get('language_preferences', {})
            for lang in ['hindi_dominant', 'english_dominant', 'balanced']:
                aggregated['language_preferences'][lang] += lang_prefs.get(lang, 0)
        
        # Normalize averages
        if total_creators > 0:
            aggregated['avg_speaking_pace'] /= total_creators
            for tone in aggregated['tone_profile']:
                aggregated['tone_profile'][tone] /= total_creators
        
        # Get top aggregated expressions and transitions
        aggregated['common_expressions'] = dict(expression_counter.most_common(10))
        aggregated['transition_phrases'] = dict(transition_counter.most_common(8))
        
        # Deduplicate and limit intro/outro patterns
        seen_intros = set()
        unique_intros = []
        for intro in aggregated['intro_patterns']:
            intro_lower = intro.lower().strip()
            if intro_lower and intro_lower not in seen_intros and len(intro) > 15:
                seen_intros.add(intro_lower)
                unique_intros.append(intro)
        unique_intros.sort(key=len, reverse=True)
        aggregated['intro_patterns'] = unique_intros[:10]
        
        seen_outros = set()
        unique_outros = []
        for outro in aggregated['outro_patterns']:
            outro_lower = outro.lower().strip()
            if outro_lower and outro_lower not in seen_outros and len(outro) > 15:
                seen_outros.add(outro_lower)
                unique_outros.append(outro)
        unique_outros.sort(key=len, reverse=True)
        aggregated['outro_patterns'] = unique_outros[:10]
        
        return aggregated
    
    def _build_aggregated_creator_context(self, topic: str, content_type: str = None, language_mix: str = "Balanced") -> str:
        """Build aggregated context from ALL creators for Auto-Select mode"""
        context_parts = []
        
        # Select diverse examples from multiple creators with language mix preference
        diverse_examples = self._select_diverse_examples_from_all(topic, content_type, language_mix)
        
        # Aggregate stats from all creators
        aggregated_stats = self._aggregate_creator_stats()
        
        # Build context header
        context_parts.append("""
=== AGGREGATED CREATOR STYLE CONTEXT (AUTO-SELECT MODE) ===
You are generating a YouTube script in an authentic blended style, inspired by multiple successful Indian YouTube creators.

BLENDED STYLE CHARACTERISTICS:
""")
        
        # Add aggregated characteristics
        avg_pace = aggregated_stats.get('avg_speaking_pace', 140)
        context_parts.append(f"- Average Speaking Pace: ~{avg_pace:.0f} words per minute")
        
        tone_profile = aggregated_stats.get('tone_profile', {})
        tone_parts = []
        if tone_profile.get('enthusiasm', 0) > 6:
            tone_parts.append("Enthusiastic")
        if tone_profile.get('technical_depth', 0) > 6:
            tone_parts.append("Technically detailed")
        if tone_profile.get('friendliness', 0) > 6:
            tone_parts.append("Friendly and approachable")
        
        if tone_parts:
            context_parts.append(f"- Tone Profile: {', '.join(tone_parts)}")
        else:
            context_parts.append("- Tone Profile: Balanced conversational style")
        
        # Language mix (most common preference)
        lang_prefs = aggregated_stats.get('language_preferences', {})
        if lang_prefs.get('hindi_dominant', 0) > lang_prefs.get('english_dominant', 0):
            lang_desc = "Hindi-dominant Hinglish (more Hindi expressions)"
        elif lang_prefs.get('english_dominant', 0) > lang_prefs.get('hindi_dominant', 0):
            lang_desc = "English-dominant Hinglish (more English words)"
        else:
            lang_desc = "Balanced Hinglish (equal mix of Hindi and English)"
        
        context_parts.append(f"- Language Style: {lang_desc}")
        context_parts.append("")
        
        # Add diverse examples from multiple creators
        if diverse_examples:
            context_parts.append("AUTHENTIC EXAMPLE OUTPUTS (from multiple successful creators):\n")
            context_parts.append("CRITICAL: These examples show STYLE and PATTERNS. Do NOT copy the text. Generate ORIGINAL content inspired by these styles.\n")
            
            for i, item in enumerate(diverse_examples[:4], 1):
                example = item['example']
                creator = item['creator']
                # Truncate to 600 chars for token management
                example_text = example['text'][:600]
                example_title = example.get('title', 'Video example')
                context_parts.append(f"EXAMPLE {i} (Creator: {creator}, from: {example_title}):")
                context_parts.append(f"{example_text}...")
                context_parts.append("")
            
            context_parts.append("STUDY THESE EXAMPLES TO UNDERSTAND:")
            context_parts.append("- Different speaking styles and flows")
            context_parts.append("- How different creators structure sentences")
            context_parts.append("- Natural mix of Hindi and English patterns")
            context_parts.append("- Various expressions and phrases used")
            context_parts.append("")
            context_parts.append("⚠️ ANTI-COPYING INSTRUCTION: Generate NEW ORIGINAL content. Do NOT copy sentences, phrases, or structure directly from examples. Use them as STYLE REFERENCE only.")
            context_parts.append("")
        
        # Add aggregated intro patterns
        intro_patterns = aggregated_stats.get('intro_patterns', [])
        if intro_patterns:
            context_parts.append("AUTHENTIC INTRO PATTERNS (blended from multiple creators):")
            for i, intro in enumerate(intro_patterns[:7], 1):
                intro_snippet = intro[:140] + "..." if len(intro) > 140 else intro
                context_parts.append(f"  {i}. {intro_snippet}")
            context_parts.append("")
        
        # Add aggregated outro patterns
        outro_patterns = aggregated_stats.get('outro_patterns', [])
        if outro_patterns:
            context_parts.append("AUTHENTIC OUTRO PATTERNS (blended from multiple creators):")
            for i, outro in enumerate(outro_patterns[:7], 1):
                outro_snippet = outro[:140] + "..." if len(outro) > 140 else outro
                context_parts.append(f"  {i}. {outro_snippet}")
            context_parts.append("")
        
        # Add aggregated common expressions
        common_expressions = aggregated_stats.get('common_expressions', {})
        transition_phrases = aggregated_stats.get('transition_phrases', {})
        
        if common_expressions or transition_phrases:
            context_parts.append("STYLE MARKERS - Common expressions and phrases from multiple creators:")
            
            if common_expressions:
                top_expressions = list(common_expressions.keys())[:10]
                context_parts.append(f"  Common Expressions: {', '.join(top_expressions)}")
            
            if transition_phrases:
                top_transitions = list(transition_phrases.keys())[:7]
                context_parts.append(f"  Transition Phrases: {', '.join(top_transitions)}")
            
            context_parts.append("")
            context_parts.append("Use these expressions naturally, but create YOUR OWN unique combinations and flow.")
            context_parts.append("")
        
        context_parts.append("CRITICAL INSTRUCTIONS FOR AUTO-SELECT MODE:")
        context_parts.append("1. Generate COMPLETELY NEW ORIGINAL content - do NOT copy from examples")
        context_parts.append("2. Blend the styles naturally - take inspiration from examples but create unique content")
        context_parts.append("3. Use expressions and patterns naturally, but in YOUR OWN way")
        context_parts.append("4. Make it authentic and fresh - as if a new creator wrote it, inspired by the best")
        context_parts.append("5. Match the speaking pace and flow style from the blended characteristics")
        context_parts.append("6. Create content that sounds natural, original, and authentically Hinglish")
        context_parts.append("")
        context_parts.append("⚠️ REMEMBER: Examples are for STYLE REFERENCE ONLY. Your script must be ORIGINAL with NEW content matching the style.")
        context_parts.append("")
        context_parts.append("=== END AGGREGATED CREATOR STYLE CONTEXT ===\n")
        
        return "\n".join(context_parts)
    
    def _get_tone_guidelines(self, tone: str) -> str:
        """Get specific guidelines for requested tone"""
        tone_guidelines = {
            'friendly_and_informative': """
- Use conversational tone
- Include friendly greetings and transitions
- Make technical concepts accessible
- Ask rhetorical questions to engage audience
""",
            'enthusiastic_and_energetic': """
- High energy language
- Use enthusiastic expressions frequently
- Create excitement about the topic
- Include dramatic emphasis on key points
""",
            'professional_and_formal': """
- More structured and formal language
- Technical accuracy is paramount
- Measured pace and tone
- Professional vocabulary choices
""",
            'casual_and_conversational': """
- Relaxed, everyday language
- Use contractions and casual expressions
- As if talking to a friend
- Include personal opinions and reactions
""",
            'dramatic_and_engaging': """
- Build suspense and excitement
- Use dramatic words and expressions
- Create a story-like narrative
- Make the audience anticipate what comes next
""",
            'technical_and_detailed': """
- Focus on specifications and technical details
- Use precise technical vocabulary
- Include comparisons and benchmarks
- Detailed explanations of features
""",
            'humorous_and_entertaining': """
- Include humor and jokes
- Use wit and clever observations
- Make the content entertaining
- Balance information with entertainment
"""
        }
        
        return tone_guidelines.get(tone, tone_guidelines['friendly_and_informative'])
    
    def _get_content_type_guidelines(self, content_type: str) -> str:
        """Get guidelines for specific content types"""
        content_guidelines = {
            'review': """
- Structure: Introduction -> Key Features -> Pros/Cons -> Performance -> Price Conclusion
- Include comparisons with similar products
- Cover practical usage scenarios
- Provide clear recommendations
""",
            'comparison': """
- Structure: Introduction -> Feature-by-feature comparison -> Performance comparison -> Value analysis -> Winner
- Create fair comparisons
- Highlight key differences
- Provide clear winner with reasoning
""",
            'guide': """
- Structure: Problem introduction -> Step-by-step solution -> Tips and tricks -> Summary
- Clear instructions with actionable steps
- Cover different scenarios
- Include troubleshooting tips
""",
            'news': """
- Structure: Breaking news -> Context and background -> Analysis -> Future implications
- Start with the most important information
- Provide context for viewers
- Analyze implications and next steps
""",
            'general': """
- Structure: Introduction -> Main content points -> Summary -> Conclusion
- Balanced mix of information and entertainment
- Engaging throughout the duration
- Clear main message or takeaway
"""
        }
        
        return content_guidelines.get(content_type, content_guidelines['general'])
    
    def _post_process_script(self, raw_script: str, target_minutes: float, target_seconds: Optional[float], hard_word_cap: int) -> Dict[str, any]:
        """Post-process the generated script for formatting, enforce hard limits, and ensure a clean ending"""
        # Short-form cinematic post-process path
        if target_seconds is not None:
            lines = [ln.strip() for ln in raw_script.split('\n') if ln.strip()]
            # Remove headings/section markers and bullets
            filtered = []
            for ln in lines:
                low = ln.lower()
                if low.startswith(('[hook', 'hook', 'intro', 'introduction', 'main', 'cta', 'outro', 'conclusion')):
                    continue
                if low.startswith(('- ', '* ')):
                    ln = ln[2:].strip()
                filtered.append(ln)

            # Join and enforce beat-cue start
            text = '\n'.join(filtered).strip()
            if not text:
                text = raw_script.strip()

            # Ensure starts with a CUT and VO
            first_line_break = text.find('\n') if '\n' in text else len(text)
            first_line = text[:first_line_break].strip()
            rest = text[first_line_break+1:] if first_line_break < len(text) else ''
            if not first_line.startswith('(CUT'):
                first_line = f"(CUT 0.0s) {first_line}"
            if 'VO:' not in first_line[:20]:
                first_line = first_line.replace('(CUT 0.0s)', '(CUT 0.0s) VO:') if '(CUT 0.0s)' in first_line else f"VO: {first_line}"

            script_text = (first_line + ('\n' + rest if rest else '')).strip()
            # Enforce word cap sentence-aware
            script_text = self._truncate_to_words_sentence_aware(script_text, hard_word_cap)
            # Prefer ending with question or clean sentence
            script_text = self._ensure_sentence_boundary(script_text)

            word_count = len(script_text.split())
            estimated_seconds = word_count / Config.SHORT_FORM_WPS
            timing_suggestions = self._create_timing_suggestions_short_form(target_seconds)
            return {
                'script': script_text,
                'word_count': word_count,
                'speaking_time': estimated_seconds / 60.0,
                'timing_suggestions': timing_suggestions,
                'pattern_markers': self._extract_applied_patterns(script_text)
            }

        # Long-form default post-process path
        # Clean and format the script
        lines = raw_script.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        formatted_script = []
        current_section = "Introduction"
        
        # Parse and reformat sections
        for line in cleaned_lines:
            if line.lower().startswith(('hook', 'intro', 'main', 'conclusion', 'outro', 'cta')):
                current_section = line.strip()
                formatted_script.append(f"\n[{current_section.upper()}]\n")
            else:
                formatted_script.append(line)
        
        script_text = "\n".join(formatted_script)
        
        # Enforce word cap with sentence-aware truncation
        script_text = self._truncate_to_words_sentence_aware(script_text, hard_word_cap)
        
        # Ensure the script ends at a sentence boundary; do not force a canned outro
        script_text = self._ensure_sentence_boundary(script_text)
        
        # Estimate word count and speaking time
        word_count = len(script_text.split())
        
        if target_seconds:
            # Short-form: estimate based on seconds
            estimated_seconds = word_count / Config.SHORT_FORM_WPS
            estimated_minutes = estimated_seconds / 60.0
        else:
            # Long-form: estimate based on WPM
            estimated_minutes = word_count / Config.SPEECH_WPM
        
        # Create timing suggestions
        if target_seconds:
            timing_suggestions = self._create_timing_suggestions_short_form(target_seconds)
        else:
            timing_suggestions = self._create_timing_suggestions(target_minutes)
        
        return {
            'script': script_text,
            'word_count': word_count,
            'speaking_time': estimated_minutes,
            'timing_suggestions': timing_suggestions,
            'pattern_markers': self._extract_applied_patterns(script_text)
        }
    
    def _parse_script_to_structure(self, script_text: str, timing_suggestions: Dict, length_minutes: float, length_seconds: Optional[float] = None) -> Dict:
        """Parse script text into structured sections with timing information"""
        import re
        from typing import List, Dict as DictType
        
        # Split script into sections
        sections = {
            'hook': {'text': '', 'start_time': 0, 'end_time': 0, 'word_count': 0},
            'intro': {'text': '', 'start_time': 0, 'end_time': 0, 'word_count': 0},
            'main_content': {'text': '', 'start_time': 0, 'end_time': 0, 'word_count': 0},
            'cta': {'text': '', 'start_time': 0, 'end_time': 0, 'word_count': 0},
            'outro': {'text': '', 'start_time': 0, 'end_time': 0, 'word_count': 0}
        }
        
        # Parse script into lines
        lines = script_text.split('\n')
        current_section = 'main_content'  # Default section
        section_texts = {'hook': [], 'intro': [], 'main_content': [], 'cta': [], 'outro': []}
        
        # Detect sections from text markers
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if any(marker in line_lower for marker in ['[hook]', 'hook:', 'hook ']):
                current_section = 'hook'
                continue
            elif any(marker in line_lower for marker in ['[intro]', 'intro:', 'introduction:', 'intro ']):
                current_section = 'intro'
                continue
            elif any(marker in line_lower for marker in ['[main]', 'main content:', 'main:', 'content:']):
                current_section = 'main_content'
                continue
            elif any(marker in line_lower for marker in ['[cta]', 'cta:', 'call to action:', 'call-to-action']):
                current_section = 'cta'
                continue
            elif any(marker in line_lower for marker in ['[outro]', 'outro:', 'conclusion:', 'ending:']):
                current_section = 'outro'
                continue
            
            # Add line to current section (skip empty lines that are section headers)
            if line.strip() and not line.strip().startswith('['):
                section_texts[current_section].append(line.strip())
        
        # Combine section texts and calculate stats
        total_duration = length_seconds if length_seconds else length_minutes * 60
        total_words = sum(len(' '.join(texts).split()) for texts in section_texts.values())
        
        # Calculate timing for each section based on word count and duration
        cumulative_time = 0
        for section_name in ['hook', 'intro', 'main_content', 'cta', 'outro']:
            section_text = ' '.join(section_texts[section_name])
            word_count = len(section_text.split())
            
            if total_words > 0:
                # Calculate section duration based on word proportion
                section_proportion = word_count / total_words if word_count > 0 else 0.05
                section_duration = total_duration * section_proportion
            else:
                section_duration = 0
            
            # Set specific timings based on section type
            if section_name == 'hook':
                start_time = 0
                end_time = min(15, section_duration)  # Hook is typically 10-15 seconds
            elif section_name == 'intro':
                start_time = cumulative_time
                end_time = start_time + min(30, section_duration)  # Intro typically 15-30 seconds
            elif section_name == 'main_content':
                start_time = cumulative_time
                end_time = min(total_duration - 30, start_time + section_duration)  # Leave room for CTA/outro
            elif section_name == 'cta':
                start_time = max(cumulative_time, total_duration - 30)
                end_time = start_time + min(20, section_duration)
            else:  # outro
                start_time = max(cumulative_time, total_duration - 15)
                end_time = total_duration
            
            sections[section_name] = {
                'text': section_text,
                'start_time': round(start_time, 1),
                'end_time': round(end_time, 1),
                'duration': round(end_time - start_time, 1),
                'word_count': word_count,
                'estimated_speaking_time_seconds': round(section_duration, 1)
            }
            
            cumulative_time = end_time
        
        # If main_content is empty, put all content there
        if not sections['main_content']['text']:
            all_text = ' '.join(section_texts['hook'] + section_texts['intro'] + 
                              section_texts['main_content'] + section_texts['cta'] + section_texts['outro'])
            if all_text:
                sections['main_content'] = {
                    'text': all_text,
                    'start_time': 15,
                    'end_time': total_duration - 30,
                    'duration': total_duration - 45,
                    'word_count': len(all_text.split()),
                    'estimated_speaking_time_seconds': total_duration - 45
                }
        
        # Build structured output
        structured = {
            'script_id': f"script_{int(time.time())}",
            'sections': sections,
            'total_duration_seconds': round(total_duration, 1),
            'total_duration_minutes': round(total_duration / 60, 2),
            'total_word_count': total_words,
            'metadata': {
                'format': 'structured_script_v1',
                'timing_format': 'seconds',
                'sections_available': list(sections.keys())
            }
        }
        
        return structured
    
    def export_structured_script(self, structured_script: Dict, format: str = 'json') -> str:
        """Export structured script in specified format (json, xml, markdown)"""
        if format.lower() == 'json':
            return json.dumps(structured_script, ensure_ascii=False, indent=2)
        elif format.lower() == 'xml':
            return self._structured_to_xml(structured_script)
        elif format.lower() == 'markdown':
            return self._structured_to_markdown(structured_script)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'xml', or 'markdown'")
    
    def _structured_to_xml(self, structured: Dict) -> str:
        """Convert structured script to XML format"""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<script>']
        
        xml_parts.append(f'  <metadata>')
        xml_parts.append(f'    <script_id>{structured["script_id"]}</script_id>')
        xml_parts.append(f'    <total_duration_seconds>{structured["total_duration_seconds"]}</total_duration_seconds>')
        xml_parts.append(f'    <total_word_count>{structured["total_word_count"]}</total_word_count>')
        xml_parts.append(f'  </metadata>')
        
        xml_parts.append('  <sections>')
        for section_name, section_data in structured['sections'].items():
            if section_data['text']:
                xml_parts.append(f'    <section name="{section_name}">')
                xml_parts.append(f'      <start_time>{section_data["start_time"]}</start_time>')
                xml_parts.append(f'      <end_time>{section_data["end_time"]}</end_time>')
                xml_parts.append(f'      <duration>{section_data["duration"]}</duration>')
                xml_parts.append(f'      <word_count>{section_data["word_count"]}</word_count>')
                xml_parts.append(f'      <text><![CDATA[{section_data["text"]}]]></text>')
                xml_parts.append(f'    </section>')
        xml_parts.append('  </sections>')
        
        xml_parts.append('</script>')
        return '\n'.join(xml_parts)
    
    def _structured_to_markdown(self, structured: Dict) -> str:
        """Convert structured script to Markdown format"""
        md_parts = [f'# Script: {structured["script_id"]}', '']
        
        md_parts.append('## Metadata')
        md_parts.append(f'- **Total Duration:** {structured["total_duration_seconds"]}s ({structured["total_duration_minutes"]} min)')
        md_parts.append(f'- **Total Words:** {structured["total_word_count"]}')
        md_parts.append('')
        
        md_parts.append('## Sections')
        md_parts.append('')
        
        for section_name, section_data in structured['sections'].items():
            if section_data['text']:
                section_title = section_name.replace('_', ' ').title()
                md_parts.append(f'### {section_title}')
                md_parts.append(f'- **Timing:** {section_data["start_time"]}s - {section_data["end_time"]}s ({section_data["duration"]}s)')
                md_parts.append(f'- **Word Count:** {section_data["word_count"]}')
                md_parts.append('')
                md_parts.append(section_data['text'])
                md_parts.append('')
        
        return '\n'.join(md_parts)

    def _truncate_to_words_sentence_aware(self, text: str, max_words: int) -> str:
        """Trim text to max_words, preferring to cut at sentence boundaries.
        Supports English (. ! ?) and Hindi danda (।) plus pipes often used as separators.
        """
        words = text.split()
        if len(words) <= max_words:
            return text
        # Build provisional trimmed text
        provisional = " ".join(words[:max_words])
        # Find last sentence boundary before cutoff
        import re
        # Include common sentence-ending punctuation
        boundary_regex = re.compile(r"[\.\!\?\|\u0964]\s")  # \u0964 is '।'
        last_boundary_idx = -1
        for match in boundary_regex.finditer(provisional):
            last_boundary_idx = match.end()
        if last_boundary_idx != -1:
            trimmed = provisional[:last_boundary_idx].strip()
        else:
            trimmed = provisional.strip()
        return trimmed

    def _ensure_sentence_boundary(self, text: str) -> str:
        """Ensure the script ends at a clean sentence boundary without injecting canned text."""
        import re
        cleaned = text.rstrip()
        # If not ending with sentence punctuation, add a period
        if not re.search(r"[\.\!\?\u0964]$", cleaned):
            cleaned = cleaned + "."
        return cleaned
    
    def _create_timing_suggestions(self, target_minutes: float) -> Dict[str, str]:
        """Create timing suggestions for long-form video production"""
        total_seconds = target_minutes * 60
        
        return {
            'hook_duration': '10-15 seconds',
            'intro_duration': '15-20 seconds', 
            'main_content_start': f'{25} seconds',
            'cta_timing': f'{total_seconds - 30} seconds',
            'outro_timing': f'{total_seconds - 15} seconds',
            'total_target': f'{target_minutes} minutes'
        }
    
    def _create_timing_suggestions_short_form(self, target_seconds: float) -> Dict[str, str]:
        """Create timing suggestions for short-form video production"""
        hook_duration = min(3, target_seconds * 0.05)  # 5% or max 3 seconds
        outro_duration = min(5, target_seconds * 0.1)  # 10% or max 5 seconds
        
        return {
            'hook_duration': f'{hook_duration:.1f} seconds',
            'intro_duration': f'{min(5, target_seconds * 0.1):.1f} seconds', 
            'main_content_start': f'{hook_duration + 2:.1f} seconds',
            'cta_timing': f'{target_seconds - outro_duration:.1f} seconds',
            'outro_timing': f'{target_seconds - outro_duration/2:.1f} seconds',
            'total_target': f'{target_seconds} seconds'
        }
    
    def _extract_applied_patterns(self, script: str) -> Dict[str, List[str]]:
        """Identify which creator patterns were applied in the script"""
        detected_patterns = {
            'hinglish_expressions': [],
            'engagement_phrases': [],
            'technical_terms': [],
            'transition_words': []
        }
        
        script_lower = script.lower()
        
        # Check for common Hinglish expressions
        hinglish_exprs = ['दोस्तों', 'भाई', 'यार', 'सुनिए', 'देखिए', 'तो यहाँ पर']
        for expr in hinglish_exprs:
            if expr in script_lower:
                detected_patterns['hinglish_expressions'].append(expr)
        
        # Check for engagement phrases
        engagement_phrases = ['subscribe', 'like', 'notification', 'bell', 'comment', 'share']
        for phrase in engagement_phrases:
            if phrase in script_lower:
                detected_patterns['engagement_phrases'].append(phrase)
        
        return detected_patterns
    
    def save_training_context(self, filepath: str):
        """Save training context to file for future use"""
        context_save_data = {
            'creator_styles': self.creator_styles,
            'training_context': self.training_context,
            'generation_config': self.generation_config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_save_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Training context saved to {filepath}")
    
    def load_training_context(self, filepath: str):
        """Load previously saved training context"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            
            self.creator_styles = context_data.get('creator_styles', {})
            self.training_context = context_data.get('training_context', {})
            
            print(f"[OK] Training context loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error loading training context: {e}")
            return False

