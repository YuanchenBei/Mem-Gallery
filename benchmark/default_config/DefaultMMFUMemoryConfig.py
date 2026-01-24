from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  
from default_config.DefaultFunctionConfig import *  

# MMFUMemory Recall Configuration
# Uses MMLMTruncation for token budget-based truncation
DEFAULT_MMFUMEMORY_RECALL = {
    'method': 'MMFUMemoryRecall',
    'truncation': DEFAULT_MMLMTRUNCATION,  # Use multimodal truncation
    'utilization': DEFAULT_MULTIMODAL_UTILIZATION,  # Use MultiModalUtilization
    'empty_memory': []
}

# MMFUMemory Store Configuration
DEFAULT_MMFUMEMORY_STORE = {
    'method': 'MMFUMemoryStore'
}

# Complete MMFUMemory Configuration
DEFAULT_MMFUMEMORY = {
    'name': 'MMFUMemory',
    'is_multimodal': True,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MMFUMEMORY_RECALL,
    'store': DEFAULT_MMFUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

