from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  
from default_config.DefaultFunctionConfig import DEFAULT_APILLM 

# UniversalRAG Routing Configuration
# Supports using APILLM (can be locally deployed model) or LocalVLLM
DEFAULT_UNIVERSALRAG_ROUTING = {
    'method': 'UniversalRAGRouting',

    # LLM configuration: can use APILLM (locally deployed OpenAI API compatible model) or LocalVLLM
    # Default uses DEFAULT_APILLM (can configure locally deployed model via base_url)
    'llm': DEFAULT_APILLM,

    # Supports 3 types: 'no', 'document', 'image'
    # supported_modalities no longer needs configuration, fixed to these 3 types

    # If LLM returns a type not in the supported list, map to the closest supported type
    'modality_mapping': {
        'no': 'no',
        'document': 'document',
        'image': 'image',
        # Fallback mapping: if LLM returns these types, map to supported types
        'paragraph': 'document',    # paragraph falls back to document
        'clip': 'image',            # clip falls back to image
        'video': 'image',            # video falls back to image
        'error': 'document',         # error falls back to document
        'text': 'document',          # text falls back to document
        'visual': 'image'            # visual falls back to image
    }
}

# UniversalRAG Storage Configuration (using GME encoder)
DEFAULT_UNIVERSALRAG_STORAGE = {
    'method': 'UniversalRAGStorage',
    'encoder': DEFAULT_GME_ENCODER,  # Use the same GME encoder as baseline
}

# UniversalRAG Retrieval Configuration
DEFAULT_UNIVERSALRAG_RETRIEVAL = {
    'method': 'UniversalRAGRetrieval',
    'top_k': DEFAULT_RETRIEVAL_TOP_K,  # Use global constant
}

# UniversalRAG Store Configuration
DEFAULT_UNIVERSALRAG_STORE = {
    'method': 'UniversalRAGStore',
}

# UniversalRAG Recall Configuration
DEFAULT_UNIVERSALRAG_RECALL = {
    'method': 'UniversalRAGRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_MULTIMODAL_UTILIZATION,  # Reuse multimodal utilization
    'top_k': DEFAULT_RETRIEVAL_TOP_K,  # Use global constant
    'empty_memory': 'None'
}

# Complete UniversalRAG Memory Configuration
DEFAULT_UNIVERSALRAGMEMORY = {
    'name': 'UniversalRAGMemory',
    'is_multimodal': True,
    'storage': DEFAULT_UNIVERSALRAG_STORAGE,
    'recall': DEFAULT_UNIVERSALRAG_RECALL,
    'store': DEFAULT_UNIVERSALRAG_STORE,
    'routing': DEFAULT_UNIVERSALRAG_ROUTING,
    'retrieval': DEFAULT_UNIVERSALRAG_RETRIEVAL,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

