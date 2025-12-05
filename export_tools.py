"""
Enhanced Export Tools for Multi-Platform Content
Export scripts, SEO content, and metadata in various formats
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time


class ExportTools:
    """Enhanced export tools for multi-platform content"""
    
    @staticmethod
    def export_script_txt(script: str, metadata: Dict, output_path: Optional[str] = None) -> str:
        """Export script as plain text file"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/script_{timestamp}.txt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""TOPIC: {metadata.get('topic', 'Unknown')}
DURATION: {metadata.get('length_minutes', metadata.get('length_seconds', 0) / 60):.1f} minutes
TONE: {metadata.get('tone_used', 'N/A')}
TARGET AUDIENCE: {metadata.get('target_audience', 'N/A')}
CONTENT TYPE: {metadata.get('content_type', 'general')}
LANGUAGE MIX: {metadata.get('language_mix', 'Balanced')}
CREATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

{script}
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    @staticmethod
    def export_script_markdown(script: str, metadata: Dict, output_path: Optional[str] = None) -> str:
        """Export script as Markdown file"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/script_{timestamp}.md"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""# {metadata.get('topic', 'YouTube Script')}

## Metadata

- **Duration**: {metadata.get('length_minutes', metadata.get('length_seconds', 0) / 60):.1f} minutes
- **Tone**: {metadata.get('tone_used', 'N/A')}
- **Target Audience**: {metadata.get('target_audience', 'N/A')}
- **Content Type**: {metadata.get('content_type', 'general')}
- **Language Mix**: {metadata.get('language_mix', 'Balanced')}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Script

{script}
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    @staticmethod
    def export_youtube_description(script: str, seo_content: Dict, metadata: Dict, output_path: Optional[str] = None) -> str:
        """Export full YouTube description with timestamps"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/youtube_description_{timestamp}.txt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get best title
        best_title = seo_content.get('titles', [{}])[0].get('title', metadata.get('topic', 'Video Title')) if seo_content.get('titles') else metadata.get('topic', 'Video Title')
        
        # Get description
        description = seo_content.get('description', f"Watch this video about {metadata.get('topic', 'this topic')}.")
        
        # Get hashtags
        hashtags = seo_content.get('hashtags', {})
        primary_tags = hashtags.get('primary_hashtags', [])
        secondary_tags = hashtags.get('secondary_hashtags', [])
        all_tags = primary_tags + secondary_tags[:5]  # Limit to 10 total
        
        content = f"""{description}

---

🔔 Subscribe for more content!
👍 Like if you found this helpful!
💬 Comment your thoughts below!

---

📱 Connect with us:
[Add your social links here]

---

#Hashtags:
{' '.join(['#' + tag.replace(' ', '') for tag in all_tags[:10]])}

---

📝 Video Details:
Topic: {metadata.get('topic', 'N/A')}
Duration: {metadata.get('length_minutes', metadata.get('length_seconds', 0) / 60):.1f} minutes
Created: {datetime.now().strftime('%Y-%m-%d')}
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    @staticmethod
    def export_instagram_caption(script: str, seo_content: Dict, metadata: Dict, output_path: Optional[str] = None) -> str:
        """Export Instagram caption"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/instagram_caption_{timestamp}.txt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get caption from SEO or generate simple one
        caption = seo_content.get('caption', f"New video about {metadata.get('topic', 'this topic')}! Check it out! 🔥")
        
        # Get hashtags
        hashtags = seo_content.get('hashtags', {})
        instagram_tags = hashtags.get('trending_hashtags', []) + hashtags.get('niche_hashtags', [])
        
        content = f"""{caption}

---

{' '.join(['#' + tag.replace(' ', '') for tag in instagram_tags[:20]])}

#YouTube #ContentCreator #Video
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    @staticmethod
    def export_tiktok_caption(script: str, seo_content: Dict, metadata: Dict, output_path: Optional[str] = None) -> str:
        """Export TikTok caption (shorter format)"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/tiktok_caption_{timestamp}.txt"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Short, punchy caption for TikTok
        topic = metadata.get('topic', 'this')
        caption = f"POV: You're watching the best {topic} video! 🎬"
        
        # Get hashtags
        hashtags = seo_content.get('hashtags', {})
        tiktok_tags = hashtags.get('trending_hashtags', []) + hashtags.get('primary_hashtags', [])
        
        content = f"""{caption}

{' '.join(['#' + tag.replace(' ', '') for tag in tiktok_tags[:10]])}

#fyp #viral #trending
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_file)
    
    @staticmethod
    def export_complete_package(script: str, seo_content: Dict, metadata: Dict, thumbnail_strategy: Optional[Dict] = None, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export complete content package (script + SEO + metadata)"""
        if not output_dir:
            timestamp = int(time.time())
            output_dir = f"output/package_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export script
        script_file = ExportTools.export_script_txt(
            script, 
            metadata, 
            str(output_path / "script.txt")
        )
        exported_files['script'] = script_file
        
        # Export markdown
        md_file = ExportTools.export_script_markdown(
            script,
            metadata,
            str(output_path / "script.md")
        )
        exported_files['markdown'] = md_file
        
        # Export YouTube description
        if seo_content:
            yt_desc = ExportTools.export_youtube_description(
                script,
                seo_content,
                metadata,
                str(output_path / "youtube_description.txt")
            )
            exported_files['youtube_description'] = yt_desc
            
            # Export Instagram caption
            ig_caption = ExportTools.export_instagram_caption(
                script,
                seo_content,
                metadata,
                str(output_path / "instagram_caption.txt")
            )
            exported_files['instagram_caption'] = ig_caption
            
            # Export TikTok caption
            tt_caption = ExportTools.export_tiktok_caption(
                script,
                seo_content,
                metadata,
                str(output_path / "tiktok_caption.txt")
            )
            exported_files['tiktok_caption'] = tt_caption
        
        # Export metadata as JSON
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': metadata,
                'seo_content': seo_content,
                'thumbnail_strategy': thumbnail_strategy,
                'exported_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        exported_files['metadata'] = str(metadata_file)
        
        return exported_files
    
    @staticmethod
    def export_batch_csv(batch_results: List[Dict], output_path: Optional[str] = None) -> str:
        """Export batch generation results as CSV"""
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output/batch_results_{timestamp}.csv"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Index', 'Topic', 'Status', 'Word Count', 'Duration', 'Tone', 
                'Success', 'Error', 'Generated At'
            ])
            
            for result in batch_results:
                metadata = result.get('metadata', {})
                writer.writerow([
                    result.get('batch_index', ''),
                    result.get('batch_topic', ''),
                    'Success' if result.get('success') else 'Failed',
                    metadata.get('estimated_word_count', 0),
                    metadata.get('length_minutes', ''),
                    metadata.get('tone_used', ''),
                    result.get('success', False),
                    result.get('error', ''),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
        
        return str(output_file)

