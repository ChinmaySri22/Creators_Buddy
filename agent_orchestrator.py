"""
Agent Orchestration System
Coordinates multiple AI agents for end-to-end content creation workflow
"""

from typing import Dict, List, Optional, Any
from ai_agents import (
    ThumbnailStrategyAgent,
    ScriptResearchAgent,
    SEOOptimizationAgent,
    QualityAssuranceAgent
)
from script_generator import ScriptGenerator
from transcript_processor import ProcessedTranscript


class AgentOrchestrator:
    """Orchestrates multiple AI agents for complete content creation workflow"""
    
    def __init__(self):
        self.thumbnail_agent = ThumbnailStrategyAgent()
        self.research_agent = ScriptResearchAgent()
        self.seo_agent = SEOOptimizationAgent()
        self.qa_agent = QualityAssuranceAgent()
        self.script_generator = None  # Will be set when available
    
    def set_script_generator(self, generator: ScriptGenerator):
        """Set the script generator for orchestrated workflows"""
        self.script_generator = generator
    
    def full_content_workflow(self, 
                             topic: str,
                             niche: Optional[str] = None,
                             script_params: Dict = None,
                             generate_thumbnail: bool = True,
                             generate_seo: bool = True) -> Dict[str, Any]:
        """
        Complete workflow: Research → Generate Script → Optimize SEO → Generate Thumbnail
        
        Returns:
            Dict with script, thumbnail strategy, SEO content, and quality scores
        """
        workflow_result = {
            'success': False,
            'steps_completed': [],
            'script': None,
            'thumbnail_strategy': None,
            'seo_content': None,
            'quality_scores': None,
            'errors': []
        }
        
        try:
            # Step 1: Research (optional)
            if niche:
                workflow_result['steps_completed'].append('research')
                try:
                    topic_suggestions = self.research_agent.suggest_topics(niche, count=5)
                    workflow_result['topic_suggestions'] = topic_suggestions
                except Exception as e:
                    workflow_result['errors'].append(f"Research step failed: {e}")
            
            # Step 2: Generate Script
            if self.script_generator and script_params:
                workflow_result['steps_completed'].append('script_generation')
                try:
                    script_result = self.script_generator.generate_script(**script_params)
                    if script_result.get('success'):
                        workflow_result['script'] = script_result.get('script')
                        workflow_result['script_metadata'] = script_result.get('metadata')
                    else:
                        workflow_result['errors'].append(f"Script generation failed: {script_result.get('error')}")
                        return workflow_result
                except Exception as e:
                    workflow_result['errors'].append(f"Script generation error: {e}")
                    return workflow_result
            else:
                workflow_result['errors'].append("Script generator not available or params missing")
                return workflow_result
            
            script_text = workflow_result['script']
            if not script_text:
                workflow_result['errors'].append("No script generated")
                return workflow_result
            
            # Step 3: Quality Assurance (parallel with SEO)
            workflow_result['steps_completed'].append('quality_assurance')
            try:
                requirements = {
                    'target_length': script_params.get('length_minutes') or script_params.get('length_seconds', 0) / 60,
                    'tone': script_params.get('tone', 'friendly_and_informative'),
                    'target_audience': script_params.get('target_audience', 'general_audience'),
                    'language_mix': script_params.get('language_mix', 'Balanced')
                }
                quality_scores = self.qa_agent.validate_script(script_text, requirements)
                workflow_result['quality_scores'] = quality_scores
            except Exception as e:
                workflow_result['errors'].append(f"Quality assurance failed: {e}")
            
            # Step 4: SEO Optimization (parallel with QA)
            if generate_seo:
                workflow_result['steps_completed'].append('seo_optimization')
                try:
                    # Generate titles
                    titles = self.seo_agent.generate_title_variations(
                        script_text, 
                        topic, 
                        count=5
                    )
                    
                    # Generate hashtags
                    hashtags = self.seo_agent.generate_hashtags(
                        script_text, 
                        topic, 
                        platform="youtube"
                    )
                    
                    # Generate description
                    description = self.seo_agent.generate_description(
                        script_text, 
                        topic, 
                        include_timestamps=True
                    )
                    
                    workflow_result['seo_content'] = {
                        'titles': titles,
                        'hashtags': hashtags,
                        'description': description
                    }
                except Exception as e:
                    workflow_result['errors'].append(f"SEO optimization failed: {e}")
            
            # Step 5: Thumbnail Strategy (after script is ready)
            if generate_thumbnail:
                workflow_result['steps_completed'].append('thumbnail_strategy')
                try:
                    thumbnail_strategy = self.thumbnail_agent.analyze_script_for_thumbnail(
                        script_text,
                        topic,
                        script_params.get('tone')
                    )
                    workflow_result['thumbnail_strategy'] = thumbnail_strategy
                except Exception as e:
                    workflow_result['errors'].append(f"Thumbnail strategy failed: {e}")
            
            workflow_result['success'] = len(workflow_result['errors']) == 0
            return workflow_result
            
        except Exception as e:
            workflow_result['errors'].append(f"Workflow error: {e}")
            return workflow_result
    
    def quick_script_workflow(self, topic: str, script_params: Dict) -> Dict[str, Any]:
        """Simplified workflow: Just generate script with QA"""
        return self.full_content_workflow(
            topic=topic,
            script_params=script_params,
            generate_thumbnail=False,
            generate_seo=False
        )
    
    def script_with_seo_workflow(self, topic: str, script_params: Dict) -> Dict[str, Any]:
        """Workflow: Generate script + SEO optimization"""
        return self.full_content_workflow(
            topic=topic,
            script_params=script_params,
            generate_thumbnail=False,
            generate_seo=True
        )

