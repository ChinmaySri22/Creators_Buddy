"""
Trends Analyzer for Content Planning
Uses YouTube Data API v3 and Google Trends (pytrends) for trend analysis
"""

import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# YouTube Data API
try:
    from googleapiclient.discovery import build
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("[WARN] google-api-python-client not installed. YouTube API features disabled.")

# Google Trends
try:
    from pytrends.request import TrendReq
    PTRENDS_AVAILABLE = True
except ImportError:
    PTRENDS_AVAILABLE = False
    print("[WARN] pytrends not installed. Google Trends features disabled.")


class TrendsAnalyzer:
    """Analyze trends for content planning"""
    
    def __init__(self, youtube_api_key: Optional[str] = None):
        self.youtube_api_key = youtube_api_key or os.getenv('YOUTUBE_API_KEY')
        self.youtube_service = None
        
        if YOUTUBE_API_AVAILABLE and self.youtube_api_key:
            try:
                self.youtube_service = build('youtube', 'v3', developerKey=self.youtube_api_key)
            except Exception as e:
                print(f"[WARN] Failed to initialize YouTube API: {e}")
        
        if PTRENDS_AVAILABLE:
            try:
                self.pytrends = TrendReq(hl='en-US', tz=360)
            except Exception as e:
                print(f"[WARN] Failed to initialize Google Trends: {e}")
                self.pytrends = None
        else:
            self.pytrends = None
    
    def get_trending_topics(self, region: str = "IN", category: str = "All", max_results: int = 25) -> List[Dict]:
        """Get trending topics from YouTube (requires API key)"""
        if not self.youtube_service:
            return []
        
        try:
            # Get trending videos
            request = self.youtube_service.videos().list(
                part="snippet,statistics,contentDetails",
                chart="mostPopular",
                regionCode=region,
                maxResults=min(max_results, 50),
                videoCategoryId=category if category != "All" else None
            )
            
            response = request.execute()
            
            trending_videos = []
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                stats = item.get('statistics', {})
                
                trending_videos.append({
                    'title': snippet.get('title', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'view_count': int(stats.get('viewCount', 0)),
                    'like_count': int(stats.get('likeCount', 0)),
                    'published_at': snippet.get('publishedAt', ''),
                    'description': snippet.get('description', '')[:200],
                    'thumbnail': snippet.get('thumbnails', {}).get('default', {}).get('url', ''),
                    'video_id': item.get('id', '')
                })
            
            return trending_videos
            
        except Exception as e:
            print(f"[ERROR] Failed to get trending topics: {e}")
            return []
    
    def search_trending_keywords(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for videos related to a keyword to understand trends"""
        if not self.youtube_service:
            return []
        
        try:
            request = self.youtube_service.search().list(
                part="snippet",
                q=query,
                type="video",
                order="viewCount",  # Sort by views to get popular content
                maxResults=min(max_results, 50),
                publishedAfter=(datetime.now() - timedelta(days=30)).isoformat() + 'Z'  # Last 30 days
            )
            
            response = request.execute()
            
            results = []
            for item in response.get('items', []):
                snippet = item.get('snippet', {})
                results.append({
                    'title': snippet.get('title', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'description': snippet.get('description', '')[:200],
                    'video_id': item.get('id', {}).get('videoId', '')
                })
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Failed to search keywords: {e}")
            return []
    
    def get_google_trends(self, keywords: List[str], timeframe: str = "today 3-m") -> Dict:
        """Get Google Trends data for keywords"""
        if not self.pytrends:
            return {}
        
        try:
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo='IN')
            
            # Get interest over time
            interest_over_time = self.pytrends.interest_over_time()
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            
            # Get trending searches
            trending_searches = self.pytrends.trending_searches(pn='india')
            
            return {
                'interest_over_time': interest_over_time.to_dict() if not interest_over_time.empty else {},
                'related_queries': {k: v.to_dict() if v is not None and not v.empty else {} for k, v in related_queries.items()},
                'trending_searches': trending_searches.tolist() if trending_searches is not None else []
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to get Google Trends: {e}")
            return {}
    
    def analyze_niche_trends(self, niche: str) -> Dict:
        """Comprehensive trend analysis for a niche"""
        results = {
            'niche': niche,
            'trending_videos': [],
            'popular_keywords': [],
            'google_trends': {},
            'insights': []
        }
        
        # Get trending videos in niche
        trending = self.search_trending_keywords(niche, max_results=20)
        results['trending_videos'] = trending
        
        # Extract popular keywords from titles
        keywords = set()
        for video in trending[:10]:
            title_words = video['title'].lower().split()
            # Filter meaningful words (3+ chars, not common words)
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
            keywords.update([w for w in title_words if len(w) > 3 and w not in common_words])
        
        results['popular_keywords'] = list(keywords)[:15]
        
        # Get Google Trends for top keywords
        if results['popular_keywords']:
            top_keywords = results['popular_keywords'][:5]
            results['google_trends'] = self.get_google_trends(top_keywords)
        
        # Generate insights
        if trending:
            avg_views = sum(int(v.get('view_count', 0)) for v in trending if 'view_count' in v) / len(trending) if trending else 0
            results['insights'].append(f"Average engagement in '{niche}' niche")
            results['insights'].append(f"Top performing content patterns identified")
        
        return results

