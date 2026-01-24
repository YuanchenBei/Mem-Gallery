from abc import ABC, abstractmethod
import re
from typing import Dict, List, Set

class BaseEntityExtractor(ABC):
    """
    Base class for entity extraction from observations.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    @abstractmethod
    def extract(self, observation) -> Dict[str, List[str]]:
        """
        Extract entities from observation.
        
        Returns:
            dict: {
                'persons': ['Lena', 'Amy', ...],
                'locations': ['park', 'university', ...],
                'objects': ['dog', 'Maltese', ...]
            }
        """
        pass

class SimpleEntityExtractor(BaseEntityExtractor):
    """
    Simple rule-based entity extractor using patterns and heuristics.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Common person name patterns
        self.person_patterns = [
            r'\b([A-Z][a-z]+)\'s\b',  # "Lena's", "Amy's"
            r'\b([A-Z][a-z]+)\b',  # Capitalized words (potential names)
        ]
        
        # Common location indicators
        self.location_indicators = [
            'park', 'university', 'school', 'home', 'office', 'restaurant',
            'museum', 'library', 'hospital', 'store', 'shop', 'market',
            'beach', 'mountain', 'city', 'town', 'country', 'state'
        ]
        
        # Common object/entity indicators
        self.object_indicators = [
            'dog', 'cat', 'pet', 'car', 'book', 'phone', 'computer',
            'bike', 'bicycle', 'house', 'apartment', 'room'
        ]
    
    def extract(self, observation) -> Dict[str, List[str]]:
        """
        Extract entities using simple patterns.
        """
        text = ""
        if isinstance(observation, dict):
            text = observation.get('text', '')
            if observation.get('image') and isinstance(observation['image'], dict):
                caption = observation['image'].get('caption', '')
                text = f"{text} {caption}".strip()
        elif isinstance(observation, str):
            text = observation
        
        if not text:
            return {'persons': [], 'locations': [], 'objects': []}
        
        text_lower = text.lower()
        
        # Extract persons
        persons = set()
        for pattern in self.person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                # Filter out common words that aren't names
                if match.lower() not in ['the', 'a', 'an', 'this', 'that', 'user', 'assistant']:
                    persons.add(match)
        
        # Extract locations
        locations = set()
        for loc in self.location_indicators:
            if loc in text_lower:
                locations.add(loc)
        # Also look for capitalized location-like words
        location_matches = re.findall(r'\b([A-Z][a-z]+)\b', text)
        for match in location_matches:
            if match.lower() in self.location_indicators or len(match) > 4:
                locations.add(match)
        
        # Extract objects
        objects = set()
        for obj in self.object_indicators:
            if obj in text_lower:
                objects.add(obj)
        # Extract breed names and specific objects
        breed_patterns = [
            r'\b(Maltese|Cairn Terrier|Toy Poodle|Scotch Terrier|Golden Retriever|Labrador)\b',
            r'\b([A-Z][a-z]+)\s+(dog|cat|pet)\b'
        ]
        for pattern in breed_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    objects.add(' '.join(match))
                else:
                    objects.add(match)
        
        return {
            'persons': list(persons),
            'locations': list(locations),
            'objects': list(objects)
        }

class LLMEntityExtractor(BaseEntityExtractor):
    """
    LLM-based entity extractor for more accurate extraction.
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
        return """Extract entities from the following content and return a JSON object with three lists: persons, locations, and objects.

Format:
{
  "persons": ["name1", "name2", ...],
  "locations": ["location1", "location2", ...],
  "objects": ["object1", "object2", ...]
}

Content:
{content}

Entities:"""
    
    def reset(self):
        if hasattr(self.llm, 'reset'):
            self.llm.reset()
    
    def extract(self, observation) -> Dict[str, List[str]]:
        """
        Extract entities using LLM.
        """
        import json
        
        # Build content string
        content_parts = []
        if isinstance(observation, dict):
            if observation.get('text'):
                content_parts.append(f"Text: {observation['text']}")
            if observation.get('image') and isinstance(observation['image'], dict):
                if observation['image'].get('caption'):
                    content_parts.append(f"Image Caption: {observation['image']['caption']}")
        elif isinstance(observation, str):
            content_parts.append(f"Text: {observation}")
        
        if not content_parts:
            return {'persons': [], 'locations': [], 'objects': []}
        
        content = '\n'.join(content_parts)
        prompt = self.extraction_prompt.format(content=content)
        
        try:
            response = self.llm.fast_run(prompt)
            # Parse JSON response
            entities = self._parse_response(response)
            return entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            # Fallback to simple extractor
            simple_extractor = SimpleEntityExtractor(self.config)
            return simple_extractor.extract(observation)
    
    def _parse_response(self, response):
        """
        Parse LLM response to extract entities.
        """
        import json
        
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                entities = json.loads(json_match.group(0))
                if isinstance(entities, dict):
                    return {
                        'persons': entities.get('persons', []),
                        'locations': entities.get('locations', []),
                        'objects': entities.get('objects', [])
                    }
            except json.JSONDecodeError:
                pass
        
        return {'persons': [], 'locations': [], 'objects': []}

