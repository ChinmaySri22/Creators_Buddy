import os
import time
import requests
import json
from typing import Optional, Dict
import google.generativeai as genai

# Import Config from config module
try:
    from config import Config
except ImportError:
    # Fallback if config not available
    class Config:
        HF_API_TOKEN = os.getenv("HF_API_TOKEN")
        HF_MODEL_ID = os.getenv("HF_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
        HF_MODEL_FALLBACK = os.getenv("HF_MODEL_FALLBACK", "stabilityai/sd-turbo")
        THUMBNAIL_DIR = os.getenv("THUMBNAIL_DIR", "thumbnails")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def _generate_prompt_with_gemini(topic: str, script_text: str, tone: Optional[str] = None) -> Dict[str, str]:
    """Use Gemini AI to analyze script and generate optimized thumbnail prompts.
    Now uses Thumbnail Strategy Agent for better results.
    """
    try:
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        
        # Use Thumbnail Strategy Agent if available
        try:
            from ai_agents import ThumbnailStrategyAgent
            agent = ThumbnailStrategyAgent()
            strategy = agent.analyze_script_for_thumbnail(script_text, topic, tone)
            
            # Use the optimized prompt from strategy
            return {
                "prompt": strategy.get("optimized_prompt", ""),
                "negative": "ugly, deformed, disfigured, poor facial features, bad anatomy, bad hands, extra limbs, extra fingers, missing fingers, low resolution, blurry, noisy, grainy, jpeg artifacts, cartoon, drawing, painting, 3d render, illustration, sketch, bad composition, poorly lit, oversaturated, desaturated, bland, boring, generic, static, plain background, duplicate, cropped, out of frame, out of focus, duplicate, tiling, multiple subjects, crowd, text, signature, watermark, logo, copyright, information, title, extra info, date, time, distorted, twisted, surreal, abstract, monochrome, black and white"
            }
        except ImportError:
            # Fallback to direct Gemini call if agent not available
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Truncate script if too long (keep first 2000 chars for analysis)
        script_snippet = script_text[:2000] if len(script_text) > 2000 else script_text
        
        prompt = f"""You are an expert at creating YouTube thumbnail prompts for AI image generation (SDXL/FLUX models).

Analyze this video script and topic, then generate an optimized prompt for creating a compelling YouTube thumbnail.

Topic: {topic}
Tone/Mood: {tone or 'dramatic'}
Script snippet:
{script_snippet}

Your task:
1. Extract key visual elements (objects, people, scenes, actions)
2. Identify emotional tone and mood
3. Suggest a compelling subject/focus for the thumbnail
4. Generate a detailed, SDXL-optimized prompt

Output format (JSON):
{{
    "visual_elements": "comma-separated list of key visual elements",
    "subject_focus": "main subject or scene to focus on",
    "emotional_tone": "emotional keywords (e.g., intense, mysterious, energetic)",
    "positive_prompt": "full optimized positive prompt for SDXL/FLUX (include: YouTube thumbnail style, cinematic quality, lighting, composition, no text)",
    "negative_prompt": "negative prompt to avoid common issues (bad quality, text, watermarks, etc.)"
}}

Guidelines:
- Focus on ONE compelling subject or scene
- Use cinematic, professional photography terms
- Include lighting, composition, and style details
- Ensure empty space on one side for text overlay
- Make it YouTube-thumbnail optimized (high contrast, bold colors, rule of thirds)
- NO text, watermarks, or logos in the image
- Keep prompts detailed but concise (under 500 words total)

Return ONLY valid JSON, no markdown formatting."""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        
        return {
            "prompt": result.get("positive_prompt", ""),
            "negative": result.get("negative_prompt", "")
        }
        
    except Exception as e:
        print(f"[WARN] Gemini prompt generation failed: {e}, falling back to basic prompt")
        return _build_fallback_prompt(topic, script_text, tone)


def _build_fallback_prompt(topic: str, script_text: str, tone: Optional[str] = None) -> Dict[str, str]:
    """Fallback prompt builder if Gemini fails"""
    # Extract basic keywords from script
    script_lower = script_text.lower()
    keywords = []
    
    # Common patterns
    if any(word in script_lower for word in ["review", "unboxing", "test"]):
        keywords.append("product showcase, detailed close-up")
    if any(word in script_lower for word in ["tutorial", "guide", "how to"]):
        keywords.append("educational, step-by-step visual")
    if any(word in script_lower for word in ["comparison", "vs", "versus"]):
        keywords.append("side-by-side comparison, split screen concept")
    
    visual_elements = ", ".join(keywords) if keywords else "dynamic, engaging scene"
    emotional_tone = tone or "captivating, intriguing"
    
    positive_prompt = f"""YouTube thumbnail, {topic}, {visual_elements}, 
highly detailed, cinematic, UHD, 8K, photorealistic,
dramatic volumetric lighting, deep shadows, bold color contrast, vibrant colors,
rule of thirds composition, shallow depth of field, sharp focus on subject, bokeh background,
epic scale, high-resolution textures, film grain, DSLR photo, 50mm lens,
{emotional_tone} mood, captivating, intriguing,
no text, no watermark, no logos, clean background with depth, empty space on one side for title overlay"""
    
    negative_prompt = """ugly, deformed, disfigured, poor facial features, bad anatomy, bad hands, extra limbs, extra fingers, missing fingers,
low resolution, blurry, noisy, grainy, jpeg artifacts, cartoon, drawing, painting, 3d render, illustration, sketch,
bad composition, poorly lit, oversaturated, desaturated, bland, boring, generic, static, plain background,
duplicate, cropped, out of frame, out of focus, duplicate, tiling, multiple subjects, crowd,
text, signature, watermark, logo, copyright, information, title, extra info, date, time,
distorted, twisted, surreal, abstract, monochrome, black and white"""
    
    return {"prompt": positive_prompt, "negative": negative_prompt}


def build_thumbnail_prompt(topic: str, script_text: str, tone: Optional[str] = None) -> Dict[str, str]:
    """Create a strong SDXL/FLUX prompt from topic/script for a YouTube thumbnail.
    Uses Gemini AI to dynamically analyze the script and generate optimized prompts.
    """
    # Try Gemini first for dynamic prompt generation
    return _generate_prompt_with_gemini(topic, script_text, tone)


def _hf_headers() -> Dict[str, str]:
    headers = {"Accept": "image/png"}
    if getattr(Config, "HF_API_TOKEN", None):
        headers["Authorization"] = f"Bearer {Config.HF_API_TOKEN}"
    return headers


def _hf_endpoint(model_id: Optional[str] = None) -> str:
    model = model_id or getattr(Config, "HF_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
    return f"https://api-inference.huggingface.co/models/{model}"


def _pollinations_fetch(prompt: str, width: int, height: int, seed: Optional[int] = None) -> Optional[bytes]:
    """Fallback: Use public Pollinations image generator (no API key)."""
    try:
        base = "https://image.pollinations.ai/prompt/"
        short_prompt = prompt[:500] # Pollinations URL can be sensitive to length
        params = {
            "w": str(width),
            "h": str(height),
        }
        if seed is not None:
            params["seed"] = str(seed)
        url = base + requests.utils.quote(short_prompt, safe="")
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200 and resp.content:
            return resp.content
        return None
    except Exception as e:
        print(f"[ERROR] Pollinations fetch failed: {e}")
        return None


def generate_thumbnail(
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    steps: int = 8, # Optimized for FLUX.1-schnell (8-12 is good balance)
    guidance: float = 3.5, # Optimized for FLUX (3-5 is good)
    model_id: Optional[str] = None,
    max_retries: int = 3, # Increased retries
    backoff_seconds: float = 5.0, # Increased backoff
) -> Optional[bytes]:
    """Generate a PNG thumbnail via HF Inference API; fallback to Pollinations (no key)."""
    # If no HF token configured, try Pollinations first for reliability
    if not getattr(Config, "HF_API_TOKEN", None):
        print("[INFO] No HF_API_TOKEN, attempting Pollinations first.")
        img = _pollinations_fetch(prompt, width, height)
        if img:
            return img
        print("[INFO] Pollinations failed, trying anonymous HF.")

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
        },
        "options": { # Added options for better control and async if needed
            "wait_for_model": True # Crucial for Hugging Face inference to wait for model to load
        }
    }
    if negative_prompt:
        payload["parameters"]["negative_prompt"] = negative_prompt

    endpoint = _hf_endpoint(model_id)
    last_err = None
    for attempt in range(max_retries + 1):
        current_endpoint = endpoint
        if attempt > 0 and model_id != Config.HF_MODEL_FALLBACK and getattr(Config, "HF_MODEL_FALLBACK", None):
            print(f"[INFO] Retrying with fallback model: {Config.HF_MODEL_FALLBACK}")
            current_endpoint = _hf_endpoint(Config.HF_MODEL_FALLBACK)

        try:
            resp = requests.post(current_endpoint, headers=_hf_headers(), json=payload, timeout=90) # Increased timeout
            if resp.status_code in (200, 201):
                return resp.content
            elif resp.status_code == 422: # Unprocessable Entity, often means invalid parameters
                print(f"[ERROR] HF Inference API returned 422 (Unprocessable Entity): {resp.text}")
                last_err = f"422: {resp.text[:200]}"
                break # Don't retry on invalid input
            elif resp.status_code in (429, 503): # Rate limited or service unavailable
                print(f"[WARN] HF Inference API returned {resp.status_code}. Retrying...")
                last_err = f"{resp.status_code}: {resp.text[:200]}"
                time.sleep(backoff_seconds * (attempt + 1))
                continue
            else: # Other HTTP errors
                print(f"[ERROR] HF Inference API error {resp.status_code}: {resp.text}")
                resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request to HF Inference API failed: {e}")
            last_err = str(e)
            time.sleep(backoff_seconds * (attempt + 1))
            
    print(f"[WARN] Thumbnail generation failed after retries for HF. Last error: {last_err}")
    # Final fallback: Pollinations (even if HF token exists and failed multiple times)
    print("[INFO] Final fallback attempt to Pollinations.")
    img = _pollinations_fetch(prompt, width, height)
    if img:
        return img
    print(f"[ERROR] All thumbnail generation attempts failed.")
    return None


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_thumbnail(image_bytes: bytes, slug: str, out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or getattr(Config, "THUMBNAIL_DIR", "thumbnails")
    ensure_dir(out_dir)
    filename = f"{slug}-{int(time.time())}.png"
    full_path = os.path.join(out_dir, filename)
    with open(full_path, "wb") as f:
        f.write(image_bytes)
    print(f"[INFO] Thumbnail saved to: {full_path}")
    return full_path