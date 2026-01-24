DEFAULT_GLOBAL_CONFIG = {
    'usable_gpu': '0,1'
}

# Global Top-K Configuration
# Unified management of all search-related top-k parameters
# Modifying these values ​​will uniformly adjust the number of searches across all memories
DEFAULT_RETRIEVAL_TOP_K = 10  # The default search term is the top-k value (used for text retrieval, multimodal retrieval, etc.).
DEFAULT_GRAPH_MAX_NODES = 10  # Maximum number of nodes traversed in a graph (for NGMemory, AUGUSTUSMemory)
DEFAULT_REFLECTION_TOP_K = 10 # Reflection retrieval of top-k (for GAReflector)

DEFAULT_OPENAI_APIKEY = '' # [Replace with your api key]
DEFAULT_OPENAI_APIBASE = '' # [Replace with your api base url]


DEFAULT_BACKBONE_PATH = '' # [Replace with your llm backbone path]
DEFAULT_GME_QWEN2_VL_7B_PATH = '' # [Replace with your GME encoder path]
DEFAULT_GME_QWEN2_VL_2B_PATH = '' # [Replace with your GME encoder path]
