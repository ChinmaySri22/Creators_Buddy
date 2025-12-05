import os
from typing import Optional, Tuple
from io import BytesIO
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance


def _load_font(preferred_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """
    Load a font, trying the preferred_path first, then assets/Poppins-Bold.ttf, then common system fonts.
    
    Note: To use Poppins-Bold.ttf, download it from Google Fonts and place it in the assets/ folder.
    """
    try:
        if preferred_path and os.path.exists(preferred_path):
            return ImageFont.truetype(preferred_path, size=size)
    except Exception:
        pass
    
    # Try assets/Poppins-Bold.ttf as first fallback (professional default for Hinglish)
    assets_font = os.path.join("assets", "Poppins-Bold.ttf")
    if os.path.exists(assets_font):
        try:
            return ImageFont.truetype(assets_font, size=size)
        except Exception:
            pass
    
    # Try common bold "YouTube" fonts
    font_fallbacks = [
        # Windows
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/arialbd.ttf",        # Arial Bold
        "C:/Windows/Fonts/BebasNeue-Regular.ttf", # Common custom font
        "C:/Windows/Fonts/Montserrat-ExtraBold.ttf", # Common custom font
        
        # macOS
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/Library/Fonts/BebasNeue.otf",
        "/Library/Fonts/Montserrat-ExtraBold.otf",
        
        # Linux
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    
    for fp in font_fallbacks:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
            
    print("[WARN] No suitable bold font found. Falling back to default.")
    print("[WARN] For best results, download 'Poppins-Bold.ttf' from Google Fonts and place it in assets/ folder.")
    print("[WARN] Alternatively, install 'Impact', 'Bebas Neue', or 'Montserrat' or provide a 'font_path'.")
    return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw, 
    text: str, 
    font_path: Optional[str], 
    target_box: Tuple[int, int, int, int], 
    max_size: int, 
    min_size: int = 24, 
    spacing: int = 8
) -> ImageFont.FreeTypeFont:
    """
    Find the largest font size that fits the text within the target_box.
    """
    left, top, right, bottom = target_box
    width = right - left
    height = bottom - top
    
    size = max_size
    while size >= min_size:
        font = _load_font(font_path, size)
        # Use multiline_textbbox for accurate line breaking and spacing
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        if w <= width and h <= height:
            return font
        size -= 2
        
    # If it never fit, return the smallest allowed size
    return _load_font(font_path, min_size)


def layout_text(
    img: Image.Image, 
    position: str = 'auto', 
    margin_ratio: float = 0.06
) -> Tuple[int, int, int, int]:
    """
    Return target box (l,t,r,b) for text. 
    'auto' reserves a column on the right, perfect for thumbnails with a left-side subject.
    """
    w, h = img.size
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)
    
    if position == 'left':
        return margin_w, margin_h, int(w * 0.55), h - margin_h
    if position == 'right' or position == 'auto':
        # Default: Use the right 55% of the screen
        return int(w * 0.45), margin_h, w - margin_w, h - margin_h
    if position == 'topbar':
        return margin_w, margin_h, w - margin_w, int(h * 0.35)
    if position == 'bottombar':
        return margin_w, int(h * 0.65), w - margin_w, h - margin_h
        
    # Fallback to 'auto'
    return int(w * 0.45), margin_h, w - margin_w, h - margin_h


def _draw_text_with_shadow_and_stroke(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill_color: Tuple[int, int, int],
    shadow_color: Tuple[int, int, int],
    stroke_color: Tuple[int, int, int],
    shadow_offset: int,
    stroke_width: int,
    spacing: int,
    align: str
):
    """
    Draw text with a drop shadow first, then a stroke, then the fill.
    This creates the cleanest, most professional look.
    """
    x, y = xy
    
    # 1. Draw the drop shadow
    shadow_xy = (x + shadow_offset, y + shadow_offset)
    draw.multiline_text(
        shadow_xy, 
        text, 
        font=font, 
        fill=shadow_color, 
        spacing=spacing, 
        align=align
    )
    
    # 2. Draw the text with stroke and fill
    draw.multiline_text(
        xy, 
        text, 
        font=font, 
        fill=fill_color, 
        stroke_width=stroke_width, 
        stroke_fill=stroke_color, 
        spacing=spacing, 
        align=align
    )


def render_text(
    image_bytes: bytes,
    text: str,
    position: str = 'auto',
    font_path: Optional[str] = None,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
    shadow_color: Tuple[int, int, int] = (0, 0, 0, 150), # Semi-transparent black shadow
    bar_color: Optional[Tuple[int, int, int]] = None,
    bar_padding: int = 20,                          
    stroke_width: int = 2,
    shadow_offset: int = 4,
    align: str = 'left',
    line_spacing: int = 8,
    max_font_size_ratio: float = 0.12
) -> bytes:
    """
    Overlay title text on image with a modern, professional look.
    Can now add a solid color bar behind the text.
    
    Args:
        image_bytes: The raw bytes of the background image.
        text: The text to render (use \n for new lines).
        position: 'auto' (right side), 'left', 'topbar', 'bottombar'.
        font_path: Path to a .ttf or .otf font file. **HIGHLY RECOMMENDED.**
        text_color: (R, G, B) tuple for the text fill.
        stroke_color: (R, G, B) tuple for the text outline.
        shadow_color: (R, G, B, A) tuple for the drop shadow.
        bar_color: (R, G, B) tuple for a solid background bar. If None, no bar is drawn.
        bar_padding: Pixels to add around the text for the background bar.
        stroke_width: Pixel width of the text outline.
        shadow_offset: Pixel offset for the drop shadow.
        align: 'left', 'center', or 'right' alignment of the text.
        line_spacing: Pixel spacing between lines of text.
        max_font_size_ratio: Max font size as a ratio of image height.
        
    Returns:
        The raw PNG bytes of the new image with text overlay.
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA') # Use RGBA mode for shadow transparency
    
    # 1. Get layout box for the overall text block
    box = layout_text(img, position)
    
    # 2. Find the best font size to fit the box
    max_size = int(img.size[1] * max_font_size_ratio)
    font = _fit_text(
        draw, 
        text, 
        font_path, 
        box, 
        max_size=max_size,
        spacing=line_spacing
    )
    
    # 3. Calculate text position (x, y) and its full bounding box
    text_virtual_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
    text_w = text_virtual_bbox[2] - text_virtual_bbox[0]
    text_h = text_virtual_bbox[3] - text_virtual_bbox[1]
    
    left, top, right, bottom = box
    area_w = right - left
    area_h = bottom - top
    
    # Calculate top-left (x, y) for the text within the `box`
    # Center vertically
    y = top + max(0, (area_h - text_h) // 2)
    
    # Position horizontally based on alignment
    if align == 'left':
        x = left
    elif align == 'right':
        x = right - text_w
    else: # 'center'
        x = left + max(0, (area_w - text_w) // 2)

    # 4. Draw the background bar IF specified (this happens BEFORE text is drawn)
    if bar_color:
        # Calculate the actual bar dimensions based on the text's final position
        bar_x1 = x - bar_padding
        bar_y1 = y - bar_padding
        bar_x2 = x + text_w + bar_padding
        bar_y2 = y + text_h + bar_padding
        
        # Clip bar coordinates to image boundaries
        bar_x1 = max(0, bar_x1)
        bar_y1 = max(0, bar_y1)
        bar_x2 = min(img.width, bar_x2)
        bar_y2 = min(img.height, bar_y2)
        
        # Draw the solid bar on the main image
        draw.rectangle([(bar_x1, bar_y1), (bar_x2, bar_y2)], fill=bar_color)

    # 5. Draw the text with all effects
    _draw_text_with_shadow_and_stroke(
        draw,
        xy=(x, y),
        text=text,
        font=font,
        fill_color=text_color,
        shadow_color=shadow_color,
        stroke_color=stroke_color,
        shadow_offset=shadow_offset,
        stroke_width=stroke_width,
        spacing=line_spacing,
        align=align
    )
    
    # 6. Export the final image
    # Convert back to RGB for saving as PNG (or keep as RGBA if preferred)
    img_rgb = img.convert('RGB')
    buf = BytesIO()
    img_rgb.save(buf, format='PNG')
    return buf.getvalue()


def enhance_thumbnail(image_bytes: bytes, 
                     enhance_contrast: float = 1.2,
                     enhance_saturation: float = 1.15,
                     enhance_sharpness: float = 1.1) -> bytes:
    """
    Post-process thumbnail for YouTube-ready quality
    
    Args:
        image_bytes: Raw image bytes
        enhance_contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        enhance_saturation: Saturation multiplier (1.0 = no change, >1.0 = more vibrant)
        enhance_sharpness: Sharpness multiplier (1.0 = no change, >1.0 = sharper)
    
    Returns:
        Enhanced image bytes
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(enhance_contrast)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(enhance_saturation)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(enhance_sharpness)
    
    # Save to bytes
    buf = BytesIO()
    img.save(buf, format='PNG', quality=95, optimize=True)
    return buf.getvalue()


def apply_youtube_optimization(image_bytes: bytes) -> bytes:
    """Apply YouTube-specific optimizations for maximum CTR"""
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Convert to numpy for advanced processing
    img_array = np.array(img)
    
    # Increase contrast slightly (YouTube thumbnails benefit from high contrast)
    img_array = np.clip(img_array * 1.1, 0, 255).astype(np.uint8)
    
    # Boost saturation in specific color ranges (reds, yellows - high CTR colors)
    hsv = np.array(Image.fromarray(img_array).convert('HSV'))
    # Boost saturation for warm colors
    saturation_mask = (hsv[:, :, 0] < 60) | (hsv[:, :, 0] > 300)  # Reds and magentas
    hsv[:, :, 1] = np.where(saturation_mask, 
                           np.clip(hsv[:, :, 1] * 1.2, 0, 255),
                           hsv[:, :, 1])
    img_array = np.array(Image.fromarray(hsv, mode='HSV').convert('RGB'))
    
    # Convert back to PIL Image
    img = Image.fromarray(img_array)
    
    # Final sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.15)
    
    # Save
    buf = BytesIO()
    img.save(buf, format='PNG', quality=95)
    return buf.getvalue()



