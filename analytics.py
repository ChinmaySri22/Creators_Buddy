"""
Analytics and Performance Tracking
Track script generation stats, usage patterns, and performance metrics
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


class Analytics:
    """Track and analyze script generation performance"""
    
    def __init__(self, analytics_file: str = "analytics.json"):
        self.analytics_file = Path(analytics_file)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load analytics data from file"""
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_data()
        return self._default_data()
    
    def _default_data(self) -> Dict:
        """Default analytics structure"""
        return {
            'total_scripts_generated': 0,
            'total_thumbnails_generated': 0,
            'scripts_by_tone': {},
            'scripts_by_audience': {},
            'scripts_by_content_type': {},
            'average_generation_time': 0,
            'success_rate': 1.0,
            'most_used_features': {},
            'daily_stats': {},
            'creator_usage': {},
            'recent_activity': []
        }
    
    def _save_data(self):
        """Save analytics data to file"""
        try:
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save analytics: {e}")
    
    def track_script_generation(self, metadata: Dict, success: bool, generation_time: float):
        """Track a script generation event"""
        self.data['total_scripts_generated'] += 1
        
        # Track by tone
        tone = metadata.get('tone_used', 'unknown')
        self.data['scripts_by_tone'][tone] = self.data['scripts_by_tone'].get(tone, 0) + 1
        
        # Track by audience
        audience = metadata.get('target_audience', 'unknown')
        self.data['scripts_by_audience'][audience] = self.data['scripts_by_audience'].get(audience, 0) + 1
        
        # Track by content type
        content_type = metadata.get('content_type', 'unknown')
        self.data['scripts_by_content_type'][content_type] = self.data['scripts_by_content_type'].get(content_type, 0) + 1
        
        # Update average generation time
        total = self.data['total_scripts_generated']
        current_avg = self.data['average_generation_time']
        self.data['average_generation_time'] = ((current_avg * (total - 1)) + generation_time) / total
        
        # Update success rate
        total_attempts = self.data.get('total_attempts', self.data['total_scripts_generated'])
        if not success:
            total_attempts += 1
        self.data['total_attempts'] = total_attempts
        self.data['success_rate'] = self.data['total_scripts_generated'] / total_attempts if total_attempts > 0 else 1.0
        
        # Track daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.data['daily_stats']:
            self.data['daily_stats'][today] = {'scripts': 0, 'thumbnails': 0, 'time_spent': 0}
        self.data['daily_stats'][today]['scripts'] += 1
        self.data['daily_stats'][today]['time_spent'] += generation_time
        
        # Add to recent activity
        activity = {
            'type': 'script_generation',
            'topic': metadata.get('topic', 'Unknown'),
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'generation_time': generation_time
        }
        self.data['recent_activity'].insert(0, activity)
        # Keep only last 50 activities
        self.data['recent_activity'] = self.data['recent_activity'][:50]
        
        self._save_data()
    
    def track_thumbnail_generation(self, success: bool):
        """Track a thumbnail generation event"""
        self.data['total_thumbnails_generated'] += 1
        
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.data['daily_stats']:
            self.data['daily_stats'][today] = {'scripts': 0, 'thumbnails': 0, 'time_spent': 0}
        self.data['daily_stats'][today]['thumbnails'] += 1
        
        self._save_data()
    
    def track_feature_usage(self, feature_name: str):
        """Track usage of a specific feature"""
        self.data['most_used_features'][feature_name] = self.data['most_used_features'].get(feature_name, 0) + 1
        self._save_data()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        # Calculate weekly stats
        week_ago = datetime.now() - timedelta(days=7)
        weekly_scripts = sum(
            stats.get('scripts', 0) 
            for date, stats in self.data['daily_stats'].items()
            if datetime.fromisoformat(date) >= week_ago
        )
        
        # Get top features
        top_features = sorted(
            self.data['most_used_features'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_scripts': self.data['total_scripts_generated'],
            'total_thumbnails': self.data['total_thumbnails_generated'],
            'weekly_scripts': weekly_scripts,
            'average_generation_time': round(self.data['average_generation_time'], 2),
            'success_rate': round(self.data['success_rate'] * 100, 1),
            'top_tones': sorted(
                self.data['scripts_by_tone'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'top_audiences': sorted(
                self.data['scripts_by_audience'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'top_content_types': sorted(
                self.data['scripts_by_content_type'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'top_features': top_features,
            'recent_activity_count': len(self.data['recent_activity'])
        }
    
    def get_daily_trends(self, days: int = 30) -> List[Dict]:
        """Get daily trends for the last N days"""
        trends = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            stats = self.data['daily_stats'].get(date, {'scripts': 0, 'thumbnails': 0, 'time_spent': 0})
            trends.append({
                'date': date,
                'scripts': stats['scripts'],
                'thumbnails': stats['thumbnails'],
                'time_spent': stats['time_spent']
            })
        return list(reversed(trends))  # Oldest first

