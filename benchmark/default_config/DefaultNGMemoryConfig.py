from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  

# NGMemory Store Configuration
DEFAULT_NGMEMORY_STORE = {
    'method': 'NGMemoryStore',
    'similarity_threshold': 0.6,  # Threshold for creating semantic similarity edges
    'max_edges_per_node': 10,  # Maximum number of semantic similarity edges per node
}

# NGMemory Recall Configuration
DEFAULT_NGMEMORY_RECALL = {
    'method': 'NGMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_MULTIMODAL_UTILIZATION,  # Use MultiModalUtilization
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'max_depth': 3,  # Maximum depth for graph traversal (used for depth_first strategy)
    'max_nodes': DEFAULT_GRAPH_MAX_NODES,  
    'traversal_threshold': 0.4,  # Minimum similarity threshold for traversal
    'empty_memory': 'None'
}

# Complete NGMemory Configuration
DEFAULT_NGMEMORY = {
    'name': 'NGMemory',
    'is_multimodal': True,
    'storage': DEFAULT_GRAPH_STORAGE,  # Use GraphStorage instead of LinearStorage
    'recall': DEFAULT_NGMEMORY_RECALL,
    'store': DEFAULT_NGMEMORY_STORE,
    'multimodal_retrieval': DEFAULT_MULTIMODAL_RETRIEVAL,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}


