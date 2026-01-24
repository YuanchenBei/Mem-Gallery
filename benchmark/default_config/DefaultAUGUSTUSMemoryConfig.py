from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  

# Concept Extractor Configuration
# Option 1: Simple keyword-based extractor (fast, no LLM)
DEFAULT_SIMPLE_CONCEPT_EXTRACTOR = {
    'method': 'SimpleConceptExtractor',
    'max_concepts': 10,  # Maximum number of concepts to extract, original 10
}

# Option 2: LLM-based extractor (slower, more accurate)
DEFAULT_LLM_CONCEPT_EXTRACTOR = {
    'method': 'LLMConceptExtractor',
    'max_concepts': 10, # original 10
    'llm_method': 'APILLM',  # or 'LocalVLLM'
    'llm': {
        'method': 'APILLM',
        'api_key': DEFAULT_OPENAI_APIKEY,
        'base_url': DEFAULT_OPENAI_APIBASE,
        'name': '',  # Default model for concept extraction. [Replace with your model path]
        #'name': 'openai/gpt-4o-mini',
        'temperature': 0.0  # Lower temperature for more consistent extraction
    },
    'extraction_prompt': None  # Use default prompt
}

# Concept-Based Retrieval Configuration
DEFAULT_CONCEPT_BASED_RETRIEVAL = {
    'method': 'ConceptBasedRetrieval',
    'topk': DEFAULT_RETRIEVAL_TOP_K, 
    'min_concept_overlap': 1,  # Minimum number of shared concepts
    'concept_weight': 1.0,  # Weight for concept-based scoring
}

# AUGUSTUSMemory Store Configuration
DEFAULT_AUGUSTUSMEMORY_STORE = {
    'method': 'AUGUSTUSMemoryStore',
    'similarity_threshold': 0.7,  # Threshold for creating semantic similarity edges
    'max_edges_per_node': 5,  # Maximum number of semantic similarity edges per node
    'min_shared_concepts': 1,  # Minimum shared concepts for concept association edges
}

# AUGUSTUSMemory Recall Configuration
DEFAULT_AUGUSTUSMEMORY_RECALL = {
    'method': 'AUGUSTUSMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_MULTIMODAL_UTILIZATION,  # Use MultiModalUtilization
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'concept_retrieval': DEFAULT_CONCEPT_BASED_RETRIEVAL,
    'max_depth': 3,  # Maximum depth for graph traversal
    'max_nodes': DEFAULT_GRAPH_MAX_NODES,  
    'traversal_threshold': 0.5,  # Minimum similarity threshold for traversal
    'embedding_weight': 0.5,  # Weight for embedding similarity in fusion
    'concept_weight': 0.5,  # Weight for concept overlap in fusion
    'empty_memory': 'None'
}

# TagGraphStorage Configuration
# TagGraphStorage uses the same config as GraphStorage, but the class itself handles concepts
DEFAULT_TAG_GRAPH_STORAGE = DEFAULT_GRAPH_STORAGE

# Complete AUGUSTUSMemory Configuration
DEFAULT_AUGUSTUSMEMORY = {
    'name': 'AUGUSTUSMemory',
    'is_multimodal': True,
    'storage': DEFAULT_TAG_GRAPH_STORAGE,  # Contextual Memory (TagGraphStorage)
    'recall': DEFAULT_AUGUSTUSMEMORY_RECALL,
    'store': DEFAULT_AUGUSTUSMEMORY_STORE,
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'concept_extractor': DEFAULT_LLM_CONCEPT_EXTRACTOR,  # Use LLM extractor as per AUGUSTUS paper
    #'concept_extractor': DEFAULT_SIMPLE_CONCEPT_EXTRACTOR,  # Comment out simple extractor
    'concept_retrieval': DEFAULT_CONCEPT_BASED_RETRIEVAL,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

