from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *

# CLIP Encoder Configuration
DEFAULT_CLIP_ENCODER = {
    'method': 'CLIPEncoder',
    'name': 'clip-vit-base-patch32',
    'dimension': 512,
    'path': '' # [Replace with your default CLIP model path (if available)]
}

# GME Qwen2-VL-7B Encoder Configuration
DEFAULT_GME_ENCODER = {
    'method': 'GMEEncoder',
    #'name': 'gme-qwen2-vl-7b',
    'name': 'gme-qwen2-vl-2b',
    #'dimension': 3584,  # GME-7B outputs 3584-dimensional embeddings
    'dimension': 1536,  # GME-2B outputs 1536-dimensional embeddings
    #'path': DEFAULT_GME_QWEN2_VL_7B_PATH
    'path': DEFAULT_GME_QWEN2_VL_2B_PATH
}


# MultiModal Retrieval Configuration
DEFAULT_MULTIMODAL_RETRIEVAL = {
    'method': 'MultiModalRetrieval',
    #'encoder': DEFAULT_CLIP_ENCODER,
    'encoder': DEFAULT_GME_ENCODER,
    #'encoder': DEFAULT_MM_EMBED_ENCODER,
    'mode': 'cosine',
    'topk': DEFAULT_RETRIEVAL_TOP_K, 
}

DEFAULT_MULTIMODAL_UTILIZATION = {
    'method': 'MultiModalUtilization',
    'prefix': '[Memory Start]',
    'suffix': '[Memory End]',
    'list_config': {
        'index': True,  
        'sep': '\n'     
    }
}

# MMMemory Store Configuration
DEFAULT_MMMEMORY_STORE = {
    'method': 'MMMemoryStore',
}

# MMMemory Recall Configuration
DEFAULT_MMMEMORY_RECALL = {
    'method': 'MMMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_MULTIMODAL_UTILIZATION,  
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'empty_memory': 'None'
}

# Complete MMMemory Configuration
DEFAULT_MMMEMORY = {
    'name': 'MMMemory',
    'is_multimodal': True,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MMMEMORY_RECALL,
    'store': DEFAULT_MMMEMORY_STORE,
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}