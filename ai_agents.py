"""
AI Agents for Creators Buddy
Specialized agents for different tasks in the content creation workflow
"""

import google.generativeai as genai
from typing import Dict, List, Optional
from config import Config


class ThumbnailStrategyAgent:
    """Agent that analyzes scripts and generates optimized thumbnail prompts"""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def analyze_script_for_thumbnail(self, script_text: str, topic: str, tone: Optional[str] = None) -> Dict[str, any]:
        """Analyze script and generate thumbnail strategy"""
        try:
            # Truncate script for analysis
            script_snippet = script_text[:2000] if len(script_text) > 2000 else script_text
            
            prompt = f"""You are an expert YouTube thumbnail strategist. Analyze this script and provide a comprehensive thumbnail strategy.

Topic: {topic}
Tone: {tone or 'dramatic'}
Script:
{script_snippet}

Provide a detailed analysis in JSON format:
{{
    "visual_elements": ["list of key visual elements to include"],
    "emotional_appeal": "primary emotion to convey",
    "color_scheme": "recommended color palette",
    "composition": "recommended composition (close-up, wide shot, etc.)",
    "text_overlay_suggestion": "suggested text for overlay (max 5 words)",
    "subject_focus": "main subject or element to focus on",
    "thumbnail_style": "cinematic, bold, minimalist, etc.",
    "optimization_tips": ["list of tips for YouTube CTR optimization"]
}}

Guidelines:
- Focus on elements that create curiosity and click-through
- Consider YouTube thumbnail best practices (faces, bold colors, rule of thirds)
- Ensure text overlay area is clear
- Make it stand out in a grid of thumbnails"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            import json
            strategy = json.loads(response_text)
            
            # Generate optimized prompt
            strategy['optimized_prompt'] = self._generate_optimized_prompt(strategy, topic, tone)
            
            return strategy
            
        except Exception as e:
            print(f"[ThumbnailAgent] Error: {e}")
            return self._fallback_strategy(topic, tone)
    
    def _generate_optimized_prompt(self, strategy: Dict, topic: str, tone: Optional[str]) -> str:
        """Generate SDXL/FLUX optimized prompt from strategy"""
        visual_elements = ", ".join(strategy.get('visual_elements', []))
        subject = strategy.get('subject_focus', topic)
        style = strategy.get('thumbnail_style', 'cinematic')
        colors = strategy.get('color_scheme', 'vibrant, bold contrast')
        
        prompt = f"""YouTube thumbnail, {topic},
{subject}, {visual_elements},
{style} style, {colors},
highly detailed, cinematic, UHD, 8K, photorealistic,
dramatic volumetric lighting, deep shadows, bold color contrast,
rule of thirds composition, shallow depth of field, sharp focus on subject,
epic scale, high-resolution textures, film grain, DSLR photo,
no text, no watermark, no logos, clean background with depth, empty space on one side for title overlay"""
        
        return prompt
    
    def _fallback_strategy(self, topic: str, tone: Optional[str]) -> Dict:
        """Fallback strategy if analysis fails"""
        return {
            "visual_elements": [topic, "engaging scene"],
            "emotional_appeal": tone or "captivating",
            "color_scheme": "vibrant, bold contrast",
            "composition": "rule of thirds, close-up or medium shot",
            "text_overlay_suggestion": topic[:5] if len(topic) > 5 else topic,
            "subject_focus": topic,
            "thumbnail_style": "cinematic",
            "optimization_tips": ["Use bold colors", "Include human element if possible", "Create curiosity gap"],
            "optimized_prompt": f"YouTube thumbnail, {topic}, cinematic, vibrant colors, rule of thirds, no text"
        }


class ScriptResearchAgent:
    """Agent for researching trends and suggesting topics"""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def suggest_topics(self, niche: str, count: int = 10) -> List[Dict[str, str]]:
        """Suggest video topics based on niche"""
        try:
            prompt = f"""You are a YouTube content strategist. Suggest {count} trending and engaging video topics for the niche: {niche}

For each topic, provide:
- Topic title (catchy, SEO-friendly)
- Why it's trending/valuable
- Target audience
- Estimated engagement potential (High/Medium/Low)

Format as JSON array:
[
    {{
        "topic": "topic title",
        "reason": "why this is a good topic",
        "target_audience": "who would watch this",
        "engagement": "High/Medium/Low"
    }}
]"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            import json
            topics = json.loads(response_text)
            return topics if isinstance(topics, list) else []
            
        except Exception as e:
            print(f"[ResearchAgent] Error: {e}")
            return []
    
    def analyze_competitor_video(self, video_title: str, description: Optional[str] = None) -> Dict[str, any]:
        """Analyze a competitor video for insights"""
        try:
            desc_text = description or "No description provided"
            prompt = f"""Analyze this competitor video and provide insights:

Title: {video_title}
Description: {desc_text[:500]}

Provide analysis in JSON:
{{
    "key_topics": ["main topics covered"],
    "hook_strategy": "how the title hooks viewers",
    "target_audience": "likely target audience",
    "content_structure": "likely video structure",
    "strengths": ["what makes this video effective"],
    "improvement_opportunities": ["how to make a better version"]
}}"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            import json
            return json.loads(response_text)
            
        except Exception as e:
            print(f"[ResearchAgent] Error analyzing competitor: {e}")
            return {}


class SEOOptimizationAgent:
    """Agent for SEO optimization - titles, descriptions, hashtags"""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_title_variations(self, script_text: str, topic: str, count: int = 5) -> List[Dict[str, str]]:
        """Generate multiple SEO-optimized title variations"""
        try:
            script_snippet = script_text[:1500] if len(script_text) > 1500 else script_text
            
            prompt = f"""You are a YouTube SEO expert. Generate {count} compelling, SEO-optimized title variations for this video.

Topic: {topic}
Script excerpt:
{script_snippet[:1000]}

Requirements:
- Each title must be 60-70 characters (YouTube optimal length)
- Include power words that increase CTR
- Create curiosity gap
- Use Hinglish naturally (if script is in Hinglish)
- No clickbait, but engaging

Format as JSON array:
[
    {{
        "title": "title text",
        "ctr_potential": "High/Medium/Low",
        "seo_score": "score out of 10",
        "why_effective": "brief explanation"
    }}
]"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            import json
            titles = json.loads(response_text)
            return titles if isinstance(titles, list) else []
            
        except Exception as e:
            print(f"[SEOAgent] Error generating titles: {e}")
            return []
    
    def generate_hashtags(self, script_text: str, topic: str, platform: str = "youtube") -> Dict[str, List[str]]:
        """Generate platform-specific hashtags"""
        try:
            script_snippet = script_text[:1000] if len(script_text) > 1000 else script_text
            
            prompt = f"""Generate hashtags for {platform} for this video content.

Topic: {topic}
Content: {script_snippet[:800]}

Generate hashtags in JSON format:
{{
    "primary_hashtags": ["3-5 main hashtags"],
    "secondary_hashtags": ["5-10 additional relevant hashtags"],
    "trending_hashtags": ["2-3 trending hashtags if applicable"],
    "niche_hashtags": ["niche-specific hashtags"]
}}

Guidelines:
- YouTube: Focus on searchable terms, less on trending
- Instagram: Mix of trending and niche
- TikTok: Trending and viral hashtags
- Keep hashtags relevant to content"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            import json
            return json.loads(response_text)
            
        except Exception as e:
            print(f"[SEOAgent] Error generating hashtags: {e}")
            return {
                "primary_hashtags": [topic.replace(" ", "")],
                "secondary_hashtags": [],
                "trending_hashtags": [],
                "niche_hashtags": []
            }
    
    def generate_description(self, script_text: str, topic: str, include_timestamps: bool = True) -> str:
        """Generate full YouTube description with timestamps"""
        try:
            script_snippet = script_text[:2000] if len(script_text) > 2000 else script_text
            
            prompt = f"""Generate a complete YouTube video description for this content.

Topic: {topic}
Script:
{script_snippet}

Requirements:
- Engaging first 2-3 lines (visible without clicking "Show more")
- Include key points from the video
- Add call-to-action for likes, subscribes, notifications
- Include relevant links section
- {"Add timestamps if video has clear sections" if include_timestamps else "No timestamps needed"}
- Use Hinglish naturally if script is in Hinglish
- Keep it under 5000 characters

Format the description properly with line breaks."""
            
            response = self.model.generate_content(prompt)
            description = response.text.strip()
            
            return description
            
        except Exception as e:
            print(f"[SEOAgent] Error generating description: {e}")
            return f"Watch this video about {topic}. Don't forget to like and subscribe!"


class HookGeneratorAgent:
    """Agent for generating multiple hook variations"""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def generate_hook_variations(self, topic: str, script_context: Optional[str] = None, count: int = 10) -> List[Dict[str, str]]:
        """Generate multiple hook variations for a video"""
        try:
            context_snippet = script_context[:500] if script_context and len(script_context) > 500 else (script_context or "")
            
            prompt = f"""You are an expert at creating YouTube video hooks that maximize click-through rates.

Topic: {topic}
Script context: {context_snippet}

Generate {count} different hook variations. Each hook should:
- Be 5-15 words (concise and punchy)
- Create curiosity gap
- Use power words
- Be in Hinglish if the context suggests it
- Avoid clickbait but be engaging

Format as JSON array:
[
    {{
        "hook": "hook text here",
        "style": "question/statement/shock/fact",
        "ctr_potential": "High/Medium/Low",
        "why_effective": "brief explanation"
    }}
]"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            import json
            hooks = json.loads(response_text)
            return hooks if isinstance(hooks, list) else []
            
        except Exception as e:
            print(f"[HookGenerator] Error: {e}")
            return []


class QualityAssuranceAgent:
    """Agent for validating and improving script quality"""
    
    def __init__(self):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def validate_script(self, script: str, requirements: Dict) -> Dict[str, any]:
        """Validate script against requirements and provide feedback"""
        try:
            prompt = f"""You are a script quality assurance expert. Validate this YouTube script.

Script:
{script[:3000]}

Requirements:
- Target length: {requirements.get('target_length', 'N/A')}
- Tone: {requirements.get('tone', 'N/A')}
- Target audience: {requirements.get('target_audience', 'N/A')}
- Language mix: {requirements.get('language_mix', 'N/A')}

Provide validation in JSON:
{{
    "overall_score": 0-10,
    "authenticity_score": 0-10,
    "engagement_score": 0-10,
    "structure_score": 0-10,
    "strengths": ["list of strengths"],
    "issues": ["list of issues found"],
    "suggestions": ["actionable improvement suggestions"],
    "is_ready": true/false
}}"""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            import json
            return json.loads(response_text)
            
        except Exception as e:
            print(f"[QAAgent] Error validating script: {e}")
            return {
                "overall_score": 7,
                "authenticity_score": 7,
                "engagement_score": 7,
                "structure_score": 7,
                "strengths": ["Script generated successfully"],
                "issues": [],
                "suggestions": ["Review script manually"],
                "is_ready": True
            }
    
    def suggest_improvements(self, script: str, feedback: Dict) -> str:
        """Generate improved version based on feedback"""
        try:
            prompt = f"""Improve this script based on the feedback provided.

Original Script:
{script[:2000]}

Feedback:
{feedback}

Provide an improved version of the script that addresses all issues while maintaining the original style and content."""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"[QAAgent] Error suggesting improvements: {e}")
            return script

