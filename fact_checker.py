"""
Fact Checker using Free Web Search APIs
Verifies claims in scripts using Google Custom Search or alternative methods
"""

import os
import requests
from typing import Dict, List, Optional
import re


class FactChecker:
    """Fact-checking agent using web search"""
    
    def __init__(self):
        # Google Custom Search API (free tier: 100 queries/day)
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        # Alternative: SerpAPI (free tier: 100/month)
        self.serpapi_key = os.getenv('SERPAPI_KEY')
    
    def check_fact(self, claim: str, context: Optional[str] = None) -> Dict[str, any]:
        """Check a single fact/claim"""
        try:
            # Extract key terms from claim
            key_terms = self._extract_key_terms(claim)
            
            # Search for verification
            search_results = self._search_web(claim, key_terms)
            
            # Analyze results
            verification = self._analyze_results(claim, search_results)
            
            return {
                'claim': claim,
                'verified': verification.get('verified', False),
                'confidence': verification.get('confidence', 'low'),
                'sources': verification.get('sources', []),
                'summary': verification.get('summary', 'Unable to verify')
            }
            
        except Exception as e:
            return {
                'claim': claim,
                'verified': None,
                'confidence': 'unknown',
                'error': str(e)
            }
    
    def check_script_facts(self, script: str, max_checks: int = 5) -> List[Dict]:
        """Extract and check facts from a script"""
        # Extract factual claims (statements with numbers, specific claims, etc.)
        claims = self._extract_claims(script, max_checks)
        
        results = []
        for claim in claims:
            result = self.check_fact(claim, script)
            results.append(result)
        
        return results
    
    def _extract_key_terms(self, claim: str) -> List[str]:
        """Extract key search terms from a claim"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b\w+\b', claim.lower())
        key_terms = [w for w in words if len(w) > 3 and w not in stop_words]
        
        return key_terms[:5]  # Top 5 terms
    
    def _extract_claims(self, script: str, max_claims: int = 5) -> List[str]:
        """Extract factual claims from script"""
        # Look for patterns that indicate facts:
        # - Numbers/statistics
        # - Specific statements
        # - Comparative statements
        
        sentences = re.split(r'[.!?।]\s+', script)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
            
            # Check for factual indicators
            has_number = bool(re.search(r'\d+', sentence))
            has_specific = any(word in sentence.lower() for word in ['is', 'are', 'has', 'have', 'contains', 'includes'])
            
            if has_number or (has_specific and len(sentence) > 30):
                claims.append(sentence)
                if len(claims) >= max_claims:
                    break
        
        return claims
    
    def _search_web(self, query: str, key_terms: List[str]) -> List[Dict]:
        """Search the web for information"""
        results = []
        
        # Try Google Custom Search first
        if self.google_api_key and self.google_cse_id:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.google_api_key,
                    'cx': self.google_cse_id,
                    'q': query[:100],  # Limit query length
                    'num': 5
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', [])[:5]:
                        results.append({
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'link': item.get('link', '')
                        })
            except Exception as e:
                print(f"[FactChecker] Google Search failed: {e}")
        
        # Fallback: Use SerpAPI if available
        if not results and self.serpapi_key:
            try:
                url = "https://serpapi.com/search"
                params = {
                    'api_key': self.serpapi_key,
                    'engine': 'google',
                    'q': query[:100],
                    'num': 5
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get('organic_results', [])[:5]:
                        results.append({
                            'title': result.get('title', ''),
                            'snippet': result.get('snippet', ''),
                            'link': result.get('link', '')
                        })
            except Exception as e:
                print(f"[FactChecker] SerpAPI failed: {e}")
        
        return results
    
    def _analyze_results(self, claim: str, search_results: List[Dict]) -> Dict:
        """Analyze search results to verify claim"""
        if not search_results:
            return {
                'verified': None,
                'confidence': 'low',
                'sources': [],
                'summary': 'No search results found'
            }
        
        # Simple verification: check if claim keywords appear in results
        claim_lower = claim.lower()
        claim_keywords = set(re.findall(r'\b\w{4,}\b', claim_lower))
        
        supporting_sources = []
        contradicting_sources = []
        
        for result in search_results:
            snippet = result.get('snippet', '').lower()
            title = result.get('title', '').lower()
            combined = snippet + ' ' + title
            
            # Count keyword matches
            matches = sum(1 for keyword in claim_keywords if keyword in combined)
            match_ratio = matches / len(claim_keywords) if claim_keywords else 0
            
            if match_ratio > 0.3:  # At least 30% keyword match
                supporting_sources.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'match_score': match_ratio
                })
            else:
                contradicting_sources.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', '')
                })
        
        # Determine verification status
        verified = len(supporting_sources) >= 2
        confidence = 'high' if len(supporting_sources) >= 3 else 'medium' if len(supporting_sources) >= 1 else 'low'
        
        summary = f"Found {len(supporting_sources)} supporting source(s)"
        if contradicting_sources:
            summary += f" and {len(contradicting_sources)} potentially contradicting source(s)"
        
        return {
            'verified': verified,
            'confidence': confidence,
            'sources': supporting_sources[:3],  # Top 3 sources
            'summary': summary
        }

