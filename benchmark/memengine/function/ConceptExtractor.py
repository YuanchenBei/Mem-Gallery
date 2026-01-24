from abc import ABC, abstractmethod
from memengine.function.LLM import APILLM, LocalVLLM
import json
import re

class BaseConceptExtractor(ABC):
    """
    Base class for concept extraction from multimodal observations.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass
    
    @abstractmethod
    def extract(self, observation):
        """
        Extract semantic concepts/labels from observation.
        
        Args:
            observation: dict with 'text' and/or 'image' fields
            
        Returns:
            list: List of concept strings (e.g., ['dog', 'walking', 'park'])
        """
        pass

class LLMConceptExtractor(BaseConceptExtractor):
    """
    Extract concepts using LLM to analyze text and image captions.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize LLM
        llm_method = getattr(config, 'llm_method', 'APILLM')
        if llm_method == 'APILLM':
            self.llm = APILLM(config.llm)
        elif llm_method == 'LocalVLLM':
            self.llm = LocalVLLM(config.llm)
        else:
            raise ValueError(f"Unsupported LLM method: {llm_method}")
        
        self.max_concepts = getattr(config, 'max_concepts', 10)
        self.extraction_prompt = getattr(config, 'extraction_prompt', None) or self._default_prompt()
    
    def _default_prompt(self):
        """
        Default prompt with few-shot examples matching AUGUSTUS paper Figure II.
        Returns tags in semicolon-separated format as per paper.
        """
        return """Extract key semantic concepts or labels from the following content. 
Return tags in the format: "tag1; tag2; tag3" (semicolon-separated).
Do not include explanations or additional text, only the tags.

Examples:
Input: "motive: gain more details about the user's pet. The image features a brown dog lying in the sand on a beach, possibly enjoying a sunny day. The dog appears relaxed and content, possibly taking a break or sunbathing."
Tags: "pet; beach; relaxed"

Input: "motive: generate more details about the user's pet. Image of a brown dog dressed as a clown, at a kids entertainment show."
Tags: "pet; clown; entertainment"

Input: "motive: understand details about the user's pet. The image features a white dog with a red collar sitting on a blue chair. The dog appears to be waiting patiently, possibly for its owner. The chair is placed near a blue wall, giving the scene a cozy atmosphere."
Tags: "pet; waiting; cozy"

Input: "motive: capture the mood about the user's cat. The photo shows a cat perched high atop a bookshelf, surrounded by various books and plants. The cat seems curious, looking directly at the camera."
Tags: "cat; curious; bookshelf"

Input: "motive: depict the evening atmosphere about the user's scene. An image capturing a serene sunset by the lake, with a couple sitting close on a picnic blanket, enjoying the view."
Tags: "sunset; lake; couple"

Input: "motive: showcase the energy about the user's city. The picture displays a bustling city street at night, lit by neon signs and busy with pedestrians."
Tags: "city; night; neon"

Input: "motive: capture the joy about the user's child. A photo of a young child laughing joyfully as they play in a garden, surrounded by flowers and butterflies."
Tags: "child; joy; garden"

Now extract concepts from:
{content}

Tags:"""
    
    def reset(self):
        if hasattr(self.llm, 'reset'):
            self.llm.reset()
    
    def extract(self, observation):
        """
        Extract concepts from observation.
        """
        # Build content string from observation
        content_parts = []
        
        if isinstance(observation, dict):
            if observation.get('text'):
                content_parts.append(f"Text: {observation['text']}")
            if observation.get('image'):
                image_info = observation['image']
                if isinstance(image_info, dict):
                    if image_info.get('caption'):
                        content_parts.append(f"Image Caption: {image_info['caption']}")
        elif isinstance(observation, str):
            content_parts.append(f"Text: {observation}")
        
        if not content_parts:
            return []
        
        content = '\n'.join(content_parts)
        
        # Use LLM to extract concepts
        prompt = self.extraction_prompt.format(content=content)
        
        try:
            response = self.llm.fast_run(prompt)
            
            # Parse JSON response
            concepts = self._parse_response(response)
            
            # Limit number of concepts
            if len(concepts) > self.max_concepts:
                concepts = concepts[:self.max_concepts]
            
            return concepts
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            # Fallback: extract simple keywords from text
            return self._fallback_extract(observation)
    
    def _parse_response(self, response):
        """
        Parse LLM response to extract concept list.
        Supports both semicolon-separated format (AUGUSTUS paper format) and JSON format.
        """
        response = response.strip()
        
        # First try semicolon-separated format (AUGUSTUS paper format)
        if ';' in response:
            # Extract tags from semicolon-separated format
            tags = [tag.strip() for tag in response.split(';')]
            tags = [tag.strip('"\'[]') for tag in tags if tag.strip()]
            if tags:
                return [str(t).lower().strip() for t in tags if t]
        
        # Try to extract JSON array from response
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Try to find JSON array
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                concepts = json.loads(json_match.group(0))
                if isinstance(concepts, list):
                    return [str(c).lower().strip() for c in concepts if c]
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse entire response as JSON
        try:
            concepts = json.loads(response)
            if isinstance(concepts, list):
                return [str(c).lower().strip() for c in concepts if c]
        except json.JSONDecodeError:
            pass
        
        # Last resort: split by comma and clean
        concepts = [c.strip().strip('"\'[]') for c in response.split(',')]
        return [c for c in concepts if c]
    
    def _fallback_extract(self, observation):
        """
        Fallback extraction: simple keyword extraction from text.
        """
        if isinstance(observation, dict):
            text = observation.get('text', '')
        elif isinstance(observation, str):
            text = observation
        else:
            return []
        
        # Simple keyword extraction (can be improved)
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Remove duplicates and limit
        keywords = list(dict.fromkeys(keywords))[:self.max_concepts]
        
        return keywords

class SimpleConceptExtractor(BaseConceptExtractor):
    """
    Simple concept extractor using keyword extraction without LLM.
    """
    def __init__(self, config):
        super().__init__(config)
        self.max_concepts = getattr(config, 'max_concepts', 10)
    
    def extract(self, observation):
        """
        Simple keyword-based concept extraction.
        """
        if isinstance(observation, dict):
            text = observation.get('text', '')
            if observation.get('image') and isinstance(observation['image'], dict):
                caption = observation['image'].get('caption', '')
                text = f"{text} {caption}".strip()
        elif isinstance(observation, str):
            text = observation
        else:
            return []
        
        if not text:
            return []
        
        # Keyword extraction
        import re
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Remove duplicates and limit
        keywords = list(dict.fromkeys(keywords))[:self.max_concepts]
        
        return keywords

