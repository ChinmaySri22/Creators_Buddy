"""
Script Validator Module
Validates generated scripts for quality, authenticity, and adherence to style guidelines
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import textstat
from langdetect import detect, DetectorFactory

# Set seed for language detection consistency
DetectorFactory.seed = 0

@dataclass
class ValidationResult:
    """Results of script validation"""
    is_valid: bool
    overall_score: float
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    metrics: Dict[str, any]
    authenticity_score: float
    readability_score: float

class ScriptValidator:
    """Validates generated YouTube scripts for quality and authenticity"""
    
    def __init__(self):
        """Initialize the validator"""
        
        # Quality thresholds
        self.thresholds = {
            'min_word_count': 100,
            'max_word_count': 5000,
            'min_hinglish_ratio': 0.2,  # At least 20% mixed language
            'max_readability_score': 85,  # Target audience should comprehend
            'min_engagement_words': 5,   # Minimum engagement phrases
            'acceptable_length_deviation': 0.3  # Within 30% of target length
        }
        
        # Style guidelines
        self.style_patterns = {
            'hinglish_expressions': [
                'दोस्तों', 'भाई', 'यार', 'सुनिए', 'देखिए', 'तो यहाँ पर',
                'बात यह है कि', 'अब मुख्य बात', 'सुनने के लिए', 'आपको पता है'
            ],
            'engagement_phrases': [
                'subscribe', 'like', 'notification', 'bell', 'comment', 'share',
                'अगर वीडियो अच्छी लगी', 'लाइक करें', 'सब्सक्राइब करें', 'बेल आइकॉन दबाएं'
            ],
            'technical_transitions': [
                'अब देखते हैं', 'यहाँ ध्यान देने वाली बात', 'अगली बात', 
                'मुख्य फीचर्स', 'परफॉर्मेंस की बात', 'डिजाइन में'
            ],
            'video_structure_markers': [
                'hook', 'intro', 'main', 'conclusion', 'outro', 'cta'
            ]
        }
        
        print("✓ Script Validator initialized")
    
    def validate_script(self, 
                       script: str, 
                       target_length_minutes: int, 
                       target_tone: str,
                       target_audience: str,
                       creator_style: Optional[str] = None,
                       content_type: str = "general") -> ValidationResult:
        """Comprehensive script validation"""
        
        print(f"Validating script ({len(script)} chars, {target_length_minutes} min target)")
        
        issues = []
        warnings = []
        suggestions = []
        
        # Basic metrics
        metrics = self._calculate_basic_metrics(script, target_length_minutes)
        
        # Validation checks
        validator_results = self._run_all_validations(
            script, target_length_minutes, target_tone, 
            target_audience, creator_style, content_type
        )
        
        issues.extend(validator_results['issues'])
        warnings.extend(validator_results['warnings'])
        suggestions.extend(validator_results['suggestions'])
        
        # Calculate scores
        overall_score = self._calculate_overall_score(metrics, validator_results)
        authenticity_score = self._calculate_authenticity_score(script, creator_style)
        readability_score = self._calculate_readability_score(script, target_audience)
        
        # Determine if script is valid
        is_valid = len(issues) == 0 and overall_score >= 0.7
        
        # Add final suggestions if needed
        if not is_valid and len(suggestions) < 3:
            suggestions.extend(self._generate_general_suggestions(script, validator_results))
        
        return ValidationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics,
            authenticity_score=authenticity_score,
            readability_score=readability_score
        )
    
    def _calculate_basic_metrics(self, script: str, target_length_minutes: int) -> Dict:
        """Calculate basic script metrics"""
        
        words = script.split()
        word_count = len(words)
        char_count = len(script)
        
        # Estimate target words based on configured speaking pace
        from config import Config
        
        # Ensure Config.SPEECH_WPM is valid
        speech_wpm = getattr(Config, 'SPEECH_WPM', 140)
        if speech_wpm is None or speech_wpm <= 0:
            speech_wpm = 140  # Default fallback
        
        # Handle None or 0 target_length_minutes
        if target_length_minutes is None or target_length_minutes <= 0:
            target_length_minutes = max(1, word_count / speech_wpm) if speech_wpm > 0 else 10  # Estimate from word count
        
        words_per_minute = target_length_minutes * Config.SPEECH_WPM
        
        # Language analysis
        language_mix = self._analyze_language_mix(script)
        
        # Readability
        readability = textstat.flesch_reading_ease(script)
        
        # Engagement markers
        engagement_score = self._count_engagement_markers(script)
        
        # Structure analysis
        structure_score = self._analyze_structure(script)
        
        # Calculate length deviation safely
        if words_per_minute > 0:
            length_deviation = abs(word_count - words_per_minute) / words_per_minute
        else:
            length_deviation = 0.0
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'words_per_minute': words_per_minute,
            'estimated_minutes': word_count / speech_wpm if speech_wpm > 0 else 0,
            'target_minutes': target_length_minutes,
            'length_deviation': length_deviation,
            'language_mix': language_mix,
            'readability_score': readability,
            'engagement_score': engagement_score,
            'structure_score': structure_score
        }
    
    def _run_all_validations(self, 
                           script: str, 
                           target_length_minutes: int,
                           target_tone: str,
                           target_audience: str,
                           creator_style: Optional[str],
                           content_type: str) -> Dict[str, List]:
        """Run all validation checks"""
        
        issues = []
        warnings = []
        suggestions = []
        
        # Length validation
        self._validate_length(script, target_length_minutes, issues, warnings, suggestions)
        
        # Language mix validation
        self._validate_language_mix(script, issues, warnings, suggestions)
        
        # Structure validation
        self._validate_structure(script, content_type, issues, warnings, suggestions)
        
        # Engagement validation
        self._validate_engagement(script, issues, warnings, suggestions)
        
        # Tone validation
        self._validate_tone(script, target_tone, issues, warnings, suggestions)
        
        # Hinglish authenticity validation
        self._validate_hinglish_authenticity(script, issues, warnings, suggestions)
        
        # Readability validation
        self._validate_readability(script, target_audience, issues, warnings, suggestions)
        
        # Creator style validation
        if creator_style:
            self._validate_creator_style(script, creator_style, issues, warnings, suggestions)
        
        return {
            'issues': issues,
            'warnings': warnings,
            'suggestions': suggestions
        }
    
    def _validate_length(self, script: str, target_minutes: int, issues: List, warnings: List, suggestions: List):
        """Validate script length"""
        
        word_count = len(script.split())
        # Handle None or invalid target_minutes
        if target_minutes is None or target_minutes <= 0:
            target_minutes = max(1, word_count / 150)  # Estimate from word count
        
        target_words = target_minutes * 150
        # Safe division
        if target_words > 0:
            deviation = abs(word_count - target_words) / target_words
        else:
            deviation = 0.0
        
        if word_count < self.thresholds['min_word_count']:
            issues.append(f"Script too short: {word_count} words (minimum: {self.thresholds['min_word_count']})")
        elif word_count > self.thresholds['max_word_count']:
            issues.append(f"Script too long: {word_count} words (maximum: {self.thresholds['max_word_count']})")
        elif deviation > self.thresholds['acceptable_length_deviation']:
            warnings.append(f"Length deviation: {deviation:.1%} from target (words: {word_count}, target: ~{target_words})")
        
        if word_count < target_words * 0.8:
            suggestions.append("Consider adding more content or examples to reach target length")
        elif word_count > target_words * 1.3:
            suggestions.append("Script is longer than target - consider condensing or splitting into series")
    
    def _validate_language_mix(self, script: str, issues: List, warnings: List, suggestions: List):
        """Validate Hinglish language mix"""
        
        language_mix = self._analyze_language_mix(script)
        mixed_ratio = language_mix['mixed_ratio']
        
        if mixed_ratio < self.thresholds['min_hinglish_ratio']:
            issues.append(f"Too little Hinglish mixing: {mixed_ratio:.1%} (minimum: {self.thresholds['min_hinglish_ratio']:.1%})")
        elif mixed_ratio > 0.8:
            warnings.append(f"Very high mixed language ratio: {mixed_ratio:.1%}")
        
        if language_mix['hindi_ratio'] > 0.8:
            suggestions.append("Consider adding more English technical terms for broader accessibility")
        elif language_mix['english_ratio'] > 0.8:
            suggestions.append("Consider adding more Hindi expressions for authenticity")
    
    def _validate_structure(self, script: str, content_type: str, issues: List, warnings: List, suggestions: List):
        """Validate video structure"""
        
        structure_score = self._analyze_structure(script)
        required_sections = self._get_required_sections(content_type)
        
        # Check for required sections
        missing_sections = []
        script_lower = script.lower()
        
        for section in required_sections:
            if section not in script_lower and not any(marker in script_lower for marker in self.style_patterns['video_structure_markers']):
                missing_sections.append(section)
        
        if missing_sections:
            warnings.append(f"Missing or unclear sections: {', '.join(missing_sections)}")
        
        if structure_score < 0.6:
            suggestions.append("Consider adding clearer section breaks or transitions")
        
        # Check for hook
        hook_indicators = ['question', '?', 'आपको', 'क्या', 'दोस्तों']
        if not any(indicator in script.lower() for indicator in hook_indicators):
            suggestions.append("Consider adding an engaging hook at the beginning")
    
    def _validate_engagement(self, script: str, issues: List, warnings: List, suggestions: List):
        """Validate engagement elements"""
        
        engagement_markers = sum(self._count_pattern_in_text(script, patterns) for patterns in self.style_patterns['engagement_phrases'])
        
        if engagement_markers < self.thresholds['min_engagement_words']:
            warnings.append(f"Low engagement elements: {engagement_markers} found (recommended: {self.thresholds['min_engagement_words']}+)")
        
        if engagement_markers == 0:
            issues.append("No call-to-action or engagement elements found")
            suggestions.append("Add subscribe, like, or comment prompts")
        elif engagement_markers < 3:
            suggestions.append("Consider adding more engagement elements like subscribe reminders")
    
    def _validate_tone(self, script: str, target_tone: str, issues: List, warnings: List, suggestions: List):
        """Validate tone adherence"""
        
        tone_markers = self._analyze_tone_markers(script)
        
        # Tone-specific validation
        if target_tone == 'enthusiastic_and_energetic':
            enthusiasm_words = ['amazing', 'fantastic', 'awesome', 'incredible', 'wow', 'बहुत', 'बहुत ही']
            has_enthusiasm = sum(1 for word in enthusiasm_words if word.lower() in script.lower())
            
            if has_enthusiasm < 3:
                suggestions.append("Add more enthusiastic expressions for energetic tone")
        
        elif target_tone == 'technical_and_detailed':
            technical_words = ['specifications', 'features', 'performance', 'benchmark', 'technology', 'स्पेक्स', 'फीचर्स']
            has_technical = sum(1 for word in technical_words if word.lower() in script.lower())
            
            if has_technical < 3:
                suggestions.append("Include more technical details and specifications")
        
        elif target_tone == 'humorous_and_entertaining':
            humor_indicators = ['funny', 'joke', 'वाह', 'कोए', 'कॉमेडी']
            has_humor = sum(1 for indicator in humor_indicators if indicator.lower() in script.lower())
            
            if has_humor < 2:
                suggestions.append("Consider adding humor or entertaining elements")
    
    def _validate_hinglish_authenticity(self, script: str, issues: List, warnings: List, suggestions: List):
        """Validate authentic Hinglish usage"""
        
        script_lower = script.lower()
        authenticity_markers = sum(self._count_pattern_in_text(script, self.style_patterns['hinglish_expressions']))
        
        if authenticity_markers < 2:
            warnings.append("Low Hinglish authenticity markers")
            suggestions.append("Consider adding more natural Hindi-English transitions")
        
        # Check for unnatural mixing patterns
        sentences = script.split('.')
        unnatural_sentences = 0
        
        for sentence in sentences[:10]:  # Check first 10 sentences
            if 'हैं' in sentence and len([c for c in sentence if c.isascii()]) > len([c for c in sentence if not c.isascii()]):
                unnatural_sentences += 1
        
        if unnatural_sentences > 3:
            suggestions.append("Some sentences may have unnatural language mixing")
    
    def _validate_readability(self, script: str, target_audience: str, issues: List, warnings: List, suggestions: List):
        """Validate readability for target audience"""
        
        readability_score = textstat.flesch_reading_ease(script)
        
        # Audience-specific readability expectations
        target_scores = {
            'general_audience': (60, 80),
            'tech_enthusiasts': (50, 70),
            'professionals': (40, 60),
            'beginners': (70, 90),
            'students': (60, 80)
        }
        
        min_score, max_score = target_scores.get(target_audience, (60, 80))
        
        if readability_score < min_score:
            warnings.append(f"High reading difficulty (score: {readability_score:.1f}, target: {min_score}+)")
            suggestions.append("Use simpler language and shorter sentences")
        elif readability_score > max_score:
            warnings.append(f"Very simple language (score: {readability_score:.1f}), might not appeal to {target_audience}")
        
        # Check sentence length
        sentences = script.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if avg_sentence_length > 25:
            suggestions.append("Consider shorter sentences for better readability")
    
    def _validate_creator_style(self, script: str, creator_style: str, issues: List, warnings: List, suggestions: List):
        """Validate adherence to specific creator style"""
        
        # This would typically check against stored creator patterns
        # For now, we'll check general style consistency
        
        style_consistency_score = self._calculate_style_consistency(script, creator_style)
        
        if style_consistency_score < 0.5:
            warnings.append(f"Low style consistency with {creator_style}")
            suggestions.append(f"Review {creator_style}'s previous videos for style patterns")
    
    def _analyze_language_mix(self, text: str) -> Dict[str, float]:
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
            return {'hindi_ratio': 0, 'english_ratio': 0, 'mixed_ratio': 0, 'total': 0}
        
        return {
            'hindi_ratio': hindi_words / total,
            'english_ratio': english_words / total,
            'mixed_ratio': mixed_words / total,
            'total': total
        }
    
    def _analyze_tone_markers(self, text: str) -> Dict[str, float]:
        """Analyze tone markers in the text"""
        text_lower = text.lower()
        text_words = text.split()
        word_count = len(text_words)
        
        enthusiasm_words = ['amazing', 'fantastic', 'awesome', 'great', 'excellent', 'incredible', 'बहुत', 'अच्छा', 'शानदार']
        technical_words = ['specifications', 'features', 'performance', 'benchmark', 'technology', 'स्पेक्स', 'फीचर्स', 'परफॉर्मेंस']
        friendly_words = ['friends', 'guys', 'bhai', 'दोस्तों', 'भाई', 'yaar', 'यार']
        
        # Safe division - avoid division by zero
        if word_count > 0:
            enthusiasm_score = sum(1 for word in enthusiasm_words if word in text_lower) / word_count * 100
            technical_score = sum(1 for word in technical_words if word in text_lower) / word_count * 100  
            friendliness_score = sum(1 for word in friendly_words if word in text_lower) / word_count * 100
        else:
            enthusiasm_score = 0.0
            technical_score = 0.0
            friendliness_score = 0.0
        
        return {
            'enthusiasm': min(enthusiasm_score, 10.0),
            'technical_depth': min(technical_score, 10.0),
            'friendliness': min(friendliness_score, 10.0)
        }
    
    def _analyze_structure(self, script: str) -> float:
        """Analyze script structure completeness"""
        
        script_lower = script.lower()
        
        # Check for structural elements
        elements_found = 0
        total_elements = 4
        
        # Hook/opener
        hook_indicators = ['दोस्तों', 'हैलो', 'hai', 'hello']
        if any(indicator in script_lower for indicator in hook_indicators):
            elements_found += 1
        
        # Main content markers
        content_markers = ['अब देखते हैं', 'मुख्य बात', 'this', 'features', 'review']
        if any(marker in script_lower for marker in content_markers):
            elements_found += 1
        
        # Engagement/call-to-action
        engagement_markers = ['like', 'subscribe', 'लाइक', 'सब्सक्राइब']
        if any(marker in script_lower for marker in engagement_markers):
            elements_found += 1
        
        # Closing/outro
        outro_markers = ['तो दोस्तों', 'see you', 'अगले वीडियो', 'bye']
        if any(marker in script_lower for marker in outro_markers):
            elements_found += 1
        
        return elements_found / total_elements
    
    def _count_engagement_markers(self, text: str) -> int:
        """Count engagement markers in text"""
        return sum(self._count_pattern_in_text(text, patterns) for patterns in self.style_patterns['engagement_phrases'])
    
    def _count_pattern_in_text(self, text: str, patterns: List[str]) -> int:
        """Count occurrences of patterns in text"""
        text_lower = text.lower()
        return sum(1 for pattern in patterns if pattern.lower() in text_lower)
    
    def _get_required_sections(self, content_type: str) -> List[str]:
        """Get required sections based on content type"""
        
        section_map = {
            'review': ['intro', 'overview', 'features', 'pros', 'cons', 'conclusion'],
            'comparison': ['intro', 'comparison', 'features', 'winner', 'conclusion'],
            'tutorial': ['intro', 'steps', 'tips', 'summary'],
            'unboxing': ['intro', 'unboxing', 'first_impressions', 'features'],
            'general': ['intro', 'main_content', 'conclusion']
        }
        
        return section_map.get(content_type, section_map['general'])
    
    def _calculate_overall_score(self, metrics: Dict, validation_results: Dict) -> float:
        """Calculate overall validation score"""
        
        # Safe division for engagement_ok
        min_engagement = self.thresholds.get('min_engagement_words', 1)
        engagement_ok = min(1, validation_results['engagement_markers'] / min_engagement) if min_engagement > 0 else 0.0
        
        # Safe division for language_mix_ok
        min_hinglish_ratio = self.thresholds.get('min_hinglish_ratio', 0.1)
        language_mix_ok = min(1, metrics['language_mix']['mixed_ratio'] / min_hinglish_ratio) if min_hinglish_ratio > 0 else 0.0
        
        # Weight different aspects
        scores = {
            'length_appropriate': max(0, 1 - metrics.get('length_deviation', 0)),
            'has_structure': metrics.get('structure_score', 0),
            'engagement_ok': engagement_ok,
            'readability_good': min(1, metrics.get('readability_score', 0) / 100) if metrics.get('readability_score') else 0,
            'language_mix_ok': language_mix_ok
        }
        
        # Weighted average
        weights = {
            'length_appropriate': 0.2,
            'has_structure': 0.25,
            'engagement_ok': 0.2,
            'readability_good': 0.15,
            'language_mix_ok': 0.2
        }
        
        weighted_score = sum(scores[key] * weights[key] for key in scores)
        return weighted_score
    
    def _calculate_authenticity_score(self, script: str, creator_style: Optional[str]) -> float:
        """Calculate authenticity score based on Hinglish usage"""
        
        authenticity_markers = sum(self._count_pattern_in_text(script, self.style_patterns['hinglish_expressions']))
        language_mix = self._analyze_language_mix(script)
        
        # Base authenticity from Hinglish expressions
        expression_score = min(1, authenticity_markers / 5)
        
        # Bonus for appropriate language mixing
        mixed_score = min(1, language_mix['mixed_ratio'] / 0.3)  # 30% mix is ideal
        
        # Penalty for pure English or pure Hindi
        if language_mix['mixed_ratio'] < 0.1:
            mixed_score *= 0.5
        
        return (expression_score + mixed_score) / 2
    
    def _calculate_readability_score(self, script: str, target_audience: str) -> float:
        """Calculate readability score"""
        
        base_score = textstat.flesch_reading_ease(script)
        
        # Adjust for target audience
        audience_adjustments = {
            'general_audience': 1.0,
            'tech_enthusiasts': 0.9,
            'professionals': 0.8,
            'beginners': 1.1,
            'students': 1.0
        }
        
        adjustment = audience_adjustments.get(target_audience, 1.0)
        adjusted_score = base_score * adjustment
        
        return min(100, max(0, adjusted_score))
    
    def _calculate_style_consistency(self, script: str, creator_style: str) -> float:
        """Calculate how consistent the script is with creator style"""
        
        # This is a simplified implementation
        # In a full system, this would compare against stored creator patterns
        
        # Generic style consistency check
        hinges_expressions = self._count_pattern_in_text(script, self.style_patterns['hinglish_expressions'])
        technical_transitions = self._count_pattern_in_text(script, self.style_patterns['technical_transitions'])
        
        total_markers = hinges_expressions + technical_transitions
        
        # Normalize based on script length
        word_count = len(script.split())
        normalized_score = total_markers / (word_count / 100)  # Markers per 100 words
        
        return min(1, normalized_score / 3)  # Scale to 0-1
    
    def _generate_general_suggestions(self, script: str, validator_results: Dict) -> List[str]:
        """Generate general improvement suggestions"""
        
        suggestions = []
        
        # Based on what was found lacking
        if validator_results.get('engagement_markers', 0) < 5:
            suggestions.append("Add more call-to-action elements")
        
        if len(script.split()) < 200:
            suggestions.append("Consider expanding content with examples or details")
        
        # Based on readability
        if textstat.flesch_reading_ease(script) < 50:
            suggestions.append("Use simpler language and shorter sentences")
        
        # Always include a few generic suggestions
        suggestions.extend([
            "Ensure natural flow between Hindi and English",
            "Add transitions between main topics",
            "Include specific examples relevant to the topic"
        ])
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def generate_quality_report(self, validation_result: ValidationResult, script_metadata: Dict) -> str:
        """Generate a detailed quality report"""
        
        report_lines = [
            f"SCRIPT QUALITY REPORT",
            f"=" * 50,
            f"",
            f"TOPIC: {script_metadata.get('topic', 'Unknown')}",
            f"DURATION: {script_metadata.get('length_minutes', 'Unknown')} minutes",
            f"TONE: {script_metadata.get('tone_used', 'Unknown')}",
            f"TARGET AUDIENCE: {script_metadata.get('target_audience', 'Unknown')}",
            f"",
            f"VALIDATION SUMMARY:",
            f"✓ Overall Score: {validation_result.overall_score:.2f}/1.0",
            f"✓ Authenticity: {validation_result.authenticity_score:.2f}/1.0", 
            f"✓ Readability: {validation_result.readability_score:.1f}/100",
            f"✓ Status: {'PASS' if validation_result.is_valid else 'ISSUES FOUND'}",
            f"",
            f"METRICS:",
            f"• Word Count: {validation_result.metrics['word_count']:,}",
            f"• Estimated Speaking Time: {validation_result.metrics['estimated_minutes']:.1f} minutes",
            f"• Language Mix: {validation_result.metrics['language_mix']['hindi_ratio']:.1%} Hindi, {validation_result.metrics['language_mix']['english_ratio']:.1%} English, {validation_result.metrics['language_mix']['mixed_ratio']:.1%} Mixed",
            f"• Structure Score: {validation_result.structure_score:.2f}/1.0",
            f""
        ]
        
        if validation_result.issues:
            report_lines.extend([
                f"CRITICAL ISSUES ({len(validation_result.issues)}):",
                *[f"• {issue}" for issue in validation_result.issues],
                f""
            ])
        
        if validation_result.warnings:
            report_lines.extend([
                f"WARNINGS ({len(validation_result.warnings)}):",
                *[f"• {warning}" for warning in validation_result.warnings],
                f""
            ])
        
        if validation_result.suggestions:
            report_lines.extend([
                f"IMPROVEMENT SUGGESTIONS ({len(validation_result.suggestions)}):",
                *[f"• {suggestion}" for suggestion in validation_result.suggestions],
                f""
            ])
        
        report_lines.extend([
            f"Generated by YouTube Script Validator",
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(report_lines)

# Utility function for external use
def validate_youtube_script(script: str, 
                          target_length_minutes: int,
                          target_tone: str = "friendly_and_informative",
                          target_audience: str = "general_audience",
                          creator_style: Optional[str] = None,
                          content_type: str = "general") -> ValidationResult:
    """Convenience function for script validation"""
    
    validator = ScriptValidator()
    return validator.validate_script(
        script, target_length_minutes, target_tone, 
        target_audience, creator_style, content_type
    )

# Add timeout import for the time module
import time

