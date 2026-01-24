from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import json
import re

class BaseFactExtractor(ABC):
    """
    Base class for fact extraction from round observations.
    Extracts structured facts (triples) similar to HippoRAG's OpenIE approach.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    @abstractmethod
    def extract(self, observation) -> List[Dict]:
        """
        Extract facts from observation.
        
        Returns:
            list: List of fact dicts, each with:
                {
                    'fact': (subject, predicate, object) tuple or string,
                    'type': 'text_fact' | 'visual_fact' | 'cross_modal_fact',
                    'source': 'text' | 'image' | 'text+image',
                    'confidence': float
                }
        """
        pass

class SimpleFactExtractor(BaseFactExtractor):
    """
    Simple rule-based fact extractor for text.
    For multimodal, extracts text facts and basic visual facts from captions.
    """
    def __init__(self, config=None):
        if config is None:
            # Create a minimal config object if none provided
            class MinimalConfig:
                pass
            config = MinimalConfig()
        super().__init__(config)
        
        # Common predicate patterns
        self.predicate_patterns = [
            (r'(\w+)\s+(is|was|are|were)\s+(.+)', 'is'),
            (r'(\w+)\s+(has|had|have)\s+(.+)', 'has'),
            (r'(\w+)\s+(went|goes|go)\s+(.+)', 'goes_to'),
            (r'(\w+)\s+(likes|liked|love|loves)\s+(.+)', 'likes'),
            (r'(\w+)\s+(lives|lived|live)\s+(.+)', 'lives_in'),
            (r'(\w+)\s+(works|worked|work)\s+(.+)', 'works_at'),
        ]
    
    def extract(self, observation) -> List[Dict]:
        """
        Extract facts from observation.
        """
        facts = []
        
        # Extract text facts
        text = ""
        if isinstance(observation, dict):
            text = observation.get('text', '')
        elif isinstance(observation, str):
            text = observation
        
        if text:
            text_facts = self._extract_text_facts(text)
            facts.extend(text_facts)
        
        # Extract visual facts from image caption
        if isinstance(observation, dict) and observation.get('image'):
            image_info = observation['image']
            if isinstance(image_info, dict) and image_info.get('caption'):
                caption = image_info['caption']
                visual_facts = self._extract_visual_facts_from_caption(caption)
                facts.extend(visual_facts)
        
        # Extract cross-modal facts if both text and image exist
        if isinstance(observation, dict) and observation.get('text') and observation.get('image'):
            cross_modal_facts = self._extract_cross_modal_facts(
                observation['text'],
                observation['image']
            )
            facts.extend(cross_modal_facts)
        
        return facts
    
    def _extract_text_facts(self, text: str) -> List[Dict]:
        """Extract facts from text using simple patterns."""
        facts = []
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern, predicate_type in self.predicate_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    subject = match.group(1).strip()
                    obj = match.group(3).strip() if len(match.groups()) >= 3 else ""
                    
                    if subject and obj:
                        facts.append({
                            'fact': (subject, predicate_type, obj),
                            'type': 'text_fact',
                            'source': 'text',
                            'confidence': 0.6,
                            'raw_text': sentence
                        })
        
        return facts
    
    def _extract_visual_facts_from_caption(self, caption: str) -> List[Dict]:
        """Extract visual facts from image caption."""
        facts = []
        
        # Simple pattern: "A [subject] [predicate] [object]"
        patterns = [
            (r'a\s+(\w+)\s+(is|are|was|were)\s+(.+)', 'is'),
            (r'a\s+(\w+)\s+(in|on|at|near)\s+(.+)', 'located_in'),
            (r'(\w+)\s+(playing|sitting|standing|running)', 'action'),
        ]
        
        for pattern, predicate_type in patterns:
            matches = re.finditer(pattern, caption.lower())
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(3).strip() if len(match.groups()) >= 3 else ""
                
                if subject:
                    facts.append({
                        'fact': (subject, predicate_type, obj),
                        'type': 'visual_fact',
                        'source': 'image',
                        'confidence': 0.5,
                        'raw_caption': caption
                    })
        
        return facts
    
    def _extract_cross_modal_facts(self, text: str, image: Dict) -> List[Dict]:
        """Extract cross-modal facts (text describes image content)."""
        facts = []
        
        caption = image.get('caption', '') if isinstance(image, dict) else ''
        
        if text and caption:
            # Find common entities/concepts between text and caption
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            caption_words = set(re.findall(r'\b\w+\b', caption.lower()))
            common_words = text_words & caption_words
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
            common_words = common_words - stop_words
            
            for word in common_words:
                if len(word) > 3:  # Only meaningful words
                    facts.append({
                        'fact': (word, 'depicted_in', 'image'),
                        'type': 'cross_modal_fact',
                        'source': 'text+image',
                        'confidence': 0.7,
                        'text': text,
                        'caption': caption
                    })
        
        return facts


class LLMFactExtractor(BaseFactExtractor):
    """
    LLM-based fact extractor for more accurate fact extraction.
    Uses LLM to extract structured facts similar to HippoRAG's OpenIE.
    """
    def __init__(self, config):
        super().__init__(config)
        
        from memengine.function.LLM import APILLM, LocalVLLM
        
        llm_method = getattr(config, 'llm_method', 'APILLM')
        if llm_method == 'APILLM':
            self.llm = APILLM(config.llm)
        elif llm_method == 'LocalVLLM':
            self.llm = LocalVLLM(config.llm)
        else:
            raise ValueError(f"Unsupported LLM method: {llm_method}")
        
        self.extraction_prompt = getattr(config, 'extraction_prompt', None) or self._default_prompt()
    
    def _default_prompt(self):
        return """Extract structured facts (subject-predicate-object triples) from the following content.
Return a JSON list of facts, each fact as:
{{"subject": "...", "predicate": "...", "object": "...", "type": "text_fact|visual_fact|cross_modal_fact"}}

Content:
{content}

Facts (JSON list):"""
    
    def reset(self):
        if hasattr(self.llm, 'reset'):
            self.llm.reset()
    
    def extract(self, observation) -> List[Dict]:
        """Extract facts using LLM."""
        facts = []
        
        # Build content string
        content_parts = []
        has_text = False
        has_image = False
        
        if isinstance(observation, dict):
            if observation.get('text'):
                content_parts.append(f"Text: {observation['text']}")
                has_text = True
            if observation.get('image'):
                image_info = observation['image']
                if isinstance(image_info, dict) and image_info.get('caption'):
                    content_parts.append(f"Image Caption: {image_info['caption']}")
                    has_image = True
        elif isinstance(observation, str):
            content_parts.append(f"Text: {observation}")
            has_text = True
        
        if not content_parts:
            return []
        
        content = '\n'.join(content_parts)
        
        # Determine fact type
        if has_text and has_image:
            fact_type = 'cross_modal_fact'
            source = 'text+image'
        elif has_image:
            fact_type = 'visual_fact'
            source = 'image'
        else:
            fact_type = 'text_fact'
            source = 'text'
        
        # Use LLM to extract facts
        prompt = self.extraction_prompt.format(content=content)
        
        try:
            response = self.llm.fast_run(prompt)
            extracted_facts = self._parse_llm_response(response, fact_type, source)
            facts.extend(extracted_facts)
        except Exception as e:
            print(f"Error extracting facts with LLM: {e}")
            # Fallback to simple extractor
            simple_extractor = SimpleFactExtractor(self.config)
            facts = simple_extractor.extract(observation)
        
        return facts
    
    def _parse_llm_response(self, response: str, default_type: str, default_source: str) -> List[Dict]:
        """Parse LLM response to extract facts."""
        facts = []
        
        try:
            # Try to parse JSON
            if response.strip().startswith('['):
                fact_list = json.loads(response)
            else:
                # Try to extract JSON from response
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    fact_list = json.loads(json_match.group())
                else:
                    return []
            
            for fact_data in fact_list:
                if isinstance(fact_data, dict):
                    subject = fact_data.get('subject', '')
                    predicate = fact_data.get('predicate', '')
                    obj = fact_data.get('object', '')
                    fact_type = fact_data.get('type', default_type)
                    
                    if subject and predicate:
                        facts.append({
                            'fact': (subject, predicate, obj),
                            'type': fact_type,
                            'source': fact_data.get('source', default_source),
                            'confidence': fact_data.get('confidence', 0.8)
                        })
        except Exception as e:
            print(f"Error parsing LLM fact response: {e}")
        
        return facts

