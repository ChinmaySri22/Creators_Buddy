"""
Batch Processing for Script Generation
Generate multiple scripts at once for content planning
"""

from typing import List, Dict, Optional
from script_generator import ScriptGenerator
from agent_orchestrator import AgentOrchestrator
import time


class BatchProcessor:
    """Process multiple script generation tasks in batch"""
    
    def __init__(self, script_generator: ScriptGenerator):
        self.script_generator = script_generator
        self.orchestrator = AgentOrchestrator()
        self.orchestrator.set_script_generator(script_generator)
    
    def generate_batch_scripts(self, 
                              topics: List[str],
                              base_params: Dict,
                              progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Generate scripts for multiple topics
        
        Args:
            topics: List of topics to generate scripts for
            base_params: Base parameters to use for all scripts (topic will be overridden)
            progress_callback: Optional callback function(completed, total, current_topic)
        
        Returns:
            List of generation results
        """
        results = []
        total = len(topics)
        
        for idx, topic in enumerate(topics, 1):
            try:
                # Override topic in params
                params = {**base_params, 'topic': topic}
                
                # Generate script
                result = self.script_generator.generate_script(**params)
                result['batch_index'] = idx
                result['batch_topic'] = topic
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(idx, total, topic)
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'batch_index': idx,
                    'batch_topic': topic
                })
        
        return results
    
    def generate_batch_with_seo(self,
                                topics: List[str],
                                base_params: Dict,
                                progress_callback: Optional[callable] = None) -> List[Dict]:
        """Generate scripts with SEO optimization for multiple topics"""
        results = []
        total = len(topics)
        
        for idx, topic in enumerate(topics, 1):
            try:
                params = {**base_params, 'topic': topic}
                
                # Use orchestrator for full workflow
                workflow_result = self.orchestrator.script_with_seo_workflow(topic, params)
                
                workflow_result['batch_index'] = idx
                workflow_result['batch_topic'] = topic
                results.append(workflow_result)
                
                if progress_callback:
                    progress_callback(idx, total, topic)
                
                time.sleep(1)  # Longer delay for full workflow
                
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'batch_index': idx,
                    'batch_topic': topic
                })
        
        return results
    
    def generate_from_calendar(self, 
                               calendar_ideas: List[Dict],
                               base_params: Dict,
                               progress_callback: Optional[callable] = None) -> List[Dict]:
        """Generate scripts for ideas from content calendar"""
        topics = [idea.get('topic') for idea in calendar_ideas if idea.get('topic')]
        return self.generate_batch_scripts(topics, base_params, progress_callback)

