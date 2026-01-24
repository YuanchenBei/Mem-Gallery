import random
from memengine.function.LLM import APILLM, LocalVLLM

class UniversalRAGRouting:
    """
    UniversalRAG routing mechanism
    Supports 3 types: 'no', 'document', 'image'
    Supports routing using APILLM (can be locally deployed model) or LocalVLLM
    """
    def __init__(self, config):
        self.config = config

        # Supports 3 types
        self.supported_modalities = ['no', 'document', 'image']

        # Simplified mapping: only keep necessary fallback mappings
        # If LLM returns a type not in the supported list, map to the closest supported type
        # Note: 'error' is not in the mapping, will be specially handled in route() method (random selection)
        default_mapping = {
            'no': 'no',
            'document': 'document',
            'image': 'image',
            'text': 'document',          # text falls back to document
            'visual': 'image'            # visual falls back to image
        }
        # Get mapping from config (may be AttributeDict or dict)
        config_mapping = getattr(config, 'modality_mapping', default_mapping)
        # Convert to regular dict, ensure .get() method can be used
        self.modality_mapping = {}

        # First start from default mapping
        for key, value in default_mapping.items():
            self.modality_mapping[key] = value

        # Then update from config mapping (if AttributeDict, use getattr; if dict, use get)
        if isinstance(config_mapping, dict):
            # Check if it's a regular dict (has get method)
            if hasattr(config_mapping, 'get') and callable(getattr(config_mapping, 'get', None)):
                # Regular dict, update directly
                self.modality_mapping.update(config_mapping)
            else:
                # AttributeDict object, access via attributes
                # Iterate over default mapping keys, get values from AttributeDict
                for key in default_mapping.keys():
                    try:
                        value = getattr(config_mapping, key, None)
                        if value is not None:
                            self.modality_mapping[key] = value
                    except:
                        pass
        else:
            # Not a dict type, try to handle as AttributeDict
            for key in default_mapping.keys():
                try:
                    value = getattr(config_mapping, key, None)
                    if value is not None:
                        self.modality_mapping[key] = value
                except:
                    pass

        # Initialize LLM (supports APILLM or LocalVLLM)
        self._init_llm(config)
        self.prompt = self._get_simplified_prompt()
    
    def _init_llm(self, config):
        """Initialize LLM (supports APILLM or LocalVLLM)"""
        # Support getting llm config from config object (AttributeDict) or dict
        if hasattr(config, 'llm'):
            llm_config = config.llm
        elif isinstance(config, dict) and 'llm' in config:
            llm_config = config['llm']
        else:
            llm_config = None

        if llm_config:
            # Handle config in dict or AttributeDict object form
            if isinstance(llm_config, dict):
                llm_method = llm_config.get('method', 'APILLM')
            else:
                # AttributeDict object
                llm_method = getattr(llm_config, 'method', 'APILLM')

            if llm_method == 'APILLM':
                self.llm = APILLM(llm_config)
            elif llm_method == 'LocalVLLM':
                self.llm = LocalVLLM(llm_config)
            else:
                raise ValueError(f"Unsupported LLM method: {llm_method}")
        else:
            # If LLM is not configured, use default APILLM config
            from default_config.DefaultFunctionConfig import DEFAULT_APILLM
            self.llm = APILLM(DEFAULT_APILLM)
    
    def _get_simplified_prompt(self):
        """
        Simplified prompt, supports 3 types
        """
        return """
Classify the following query into one of three categories: [No, Document, Image], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:
- No: The query can be answered directly with common knowledge, reasoning, or computation without external data. No retrieval is needed.
- Document: The query requires retrieving information from text sources. This includes factual descriptions, explanations, summaries, or multi-hop reasoning that needs information from stored memories.
- Image: The query focuses on visual aspects like appearances, structures, spatial relationships, or requires visual information from images.

Examples:
1. "What is the capital of France?" → No
2. "Solve 12 × 8." → No
3. "What is the birth date of Alan Turing?" → Document
4. "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?" → Document
5. "Who played a key role in the development of the iPhone?" → Document
6. "Describe the appearance of a blue whale." → Image
7. "Describe the structure of the Eiffel Tower." → Image
8. "What does the image show?" → Image

Classify the following query: {query}
Provide only the category (No, Document, or Image).
"""
    
    def reset(self):
        pass
    
    def route(self, query):
        """
        Route query to appropriate retrieval type

        Args:
            query: Query (dict or str)

        Returns:
            str: 'no', 'document', or 'image'
        """
        query_text = query.get('text', '') if isinstance(query, dict) else str(query)

        # Use LLM for routing
        raw_modality = self._route_with_llm(query_text)

        # Align with official implementation: if modality is 'error', randomly select a supported modality
        raw_modality_lower = raw_modality.lower()
        if raw_modality_lower == 'error':
            # Randomly select a supported modality (align with official implementation)
            mapped_modality = random.choice(self.supported_modalities)
            return mapped_modality

        # Map to supported 3 types (self.modality_mapping is already a regular dict)
        mapped_modality = self.modality_mapping.get(raw_modality_lower, 'document')

        # Ensure in supported list
        if mapped_modality not in self.supported_modalities:
            mapped_modality = 'document'  # Default fallback to document

        return mapped_modality
    
    def _route_with_llm(self, query_text):
        """Use LLM for routing (supports APILLM or LocalVLLM)"""
        prompt_text = self.prompt.format(query=query_text)
        try:
            # Use LLM's fast_run method
            response = self.llm.fast_run(prompt_text)
            retrieval = response.strip().lower()

            # Normalize output (handle possible variants)
            if 'no' in retrieval or retrieval == 'none':
                return 'no'
            elif 'document' in retrieval or 'text' in retrieval:
                return 'document'
            elif 'image' in retrieval or 'visual' in retrieval:
                return 'image'
            else:
                # If unrecognized, return 'error' (align with official implementation, will randomly select in route())
                return 'error'
        except Exception as e:
            print(f"LLM routing error: {e}")
            # Align with official implementation: return 'error' on exception, will randomly select in route()
            return 'error'

