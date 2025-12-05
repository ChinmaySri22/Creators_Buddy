"""
Content Calendar for Video Planning
Simple file-based calendar system for scheduling video ideas
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class VideoIdea:
    """Represents a video idea in the content calendar"""
    id: str
    topic: str
    scheduled_date: str  # ISO format date
    status: str  # 'planned', 'script_ready', 'filming', 'editing', 'published'
    niche: Optional[str] = None
    notes: Optional[str] = None
    script_generated: bool = False
    thumbnail_generated: bool = False
    seo_ready: bool = False
    created_at: Optional[str] = None


class ContentCalendar:
    """Manage content calendar for video planning"""
    
    def __init__(self, calendar_file: str = "content_calendar.json"):
        self.calendar_file = Path(calendar_file)
        self.ideas: List[VideoIdea] = []
        self._load_calendar()
    
    def _load_calendar(self):
        """Load calendar from file"""
        if self.calendar_file.exists():
            try:
                with open(self.calendar_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.ideas = [VideoIdea(**item) for item in data.get('ideas', [])]
            except Exception as e:
                print(f"[WARN] Failed to load calendar: {e}")
                self.ideas = []
        else:
            self.ideas = []
    
    def _save_calendar(self):
        """Save calendar to file"""
        try:
            data = {
                'ideas': [asdict(idea) for idea in self.ideas],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.calendar_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save calendar: {e}")
    
    def add_idea(self, topic: str, scheduled_date: str, niche: Optional[str] = None, notes: Optional[str] = None) -> VideoIdea:
        """Add a new video idea to the calendar"""
        idea_id = f"idea_{int(datetime.now().timestamp())}"
        idea = VideoIdea(
            id=idea_id,
            topic=topic,
            scheduled_date=scheduled_date,
            status='planned',
            niche=niche,
            notes=notes,
            created_at=datetime.now().isoformat()
        )
        self.ideas.append(idea)
        self._save_calendar()
        return idea
    
    def update_idea(self, idea_id: str, **updates) -> bool:
        """Update an existing video idea"""
        for idea in self.ideas:
            if idea.id == idea_id:
                for key, value in updates.items():
                    if hasattr(idea, key):
                        setattr(idea, key, value)
                self._save_calendar()
                return True
        return False
    
    def delete_idea(self, idea_id: str) -> bool:
        """Delete a video idea"""
        original_count = len(self.ideas)
        self.ideas = [idea for idea in self.ideas if idea.id != idea_id]
        if len(self.ideas) < original_count:
            self._save_calendar()
            return True
        return False
    
    def get_ideas_by_date_range(self, start_date: str, end_date: str) -> List[VideoIdea]:
        """Get ideas within a date range"""
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        return [
            idea for idea in self.ideas
            if start <= datetime.fromisoformat(idea.scheduled_date) <= end
        ]
    
    def get_upcoming_ideas(self, days: int = 30) -> List[VideoIdea]:
        """Get upcoming ideas in the next N days"""
        today = datetime.now().date()
        end_date = (today + timedelta(days=days)).isoformat()
        start_date = today.isoformat()
        
        return self.get_ideas_by_date_range(start_date, end_date)
    
    def get_ideas_by_status(self, status: str) -> List[VideoIdea]:
        """Get ideas by status"""
        return [idea for idea in self.ideas if idea.status == status]
    
    def get_ideas_by_niche(self, niche: str) -> List[VideoIdea]:
        """Get ideas by niche"""
        return [idea for idea in self.ideas if idea.niche and idea.niche.lower() == niche.lower()]
    
    def get_statistics(self) -> Dict:
        """Get calendar statistics"""
        total = len(self.ideas)
        by_status = {}
        for idea in self.ideas:
            by_status[idea.status] = by_status.get(idea.status, 0) + 1
        
        upcoming = len(self.get_upcoming_ideas(30))
        
        return {
            'total_ideas': total,
            'by_status': by_status,
            'upcoming_30_days': upcoming,
            'script_ready': sum(1 for idea in self.ideas if idea.script_generated),
            'thumbnail_ready': sum(1 for idea in self.ideas if idea.thumbnail_generated),
            'seo_ready': sum(1 for idea in self.ideas if idea.seo_ready)
        }

