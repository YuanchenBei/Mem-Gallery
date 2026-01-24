from default_config.DefaultFunctionConfig import *

# ----- Recall -----
DEFAULT_FUMEMORY_RECALL = {
    'method': 'FUMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'empty_memory': 'None'
}

DEFAULT_STMEMORY_RECALL = {
    'method': 'STMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'time_retrieval': DEFAULT_TIME_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_LTMEMORY_RECALL = {
    'method': 'LTMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_GAMEMORY_RECALL = {
    'method': 'GAMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'time_retrieval': {
        'method': 'TimeRetrieval',
        'mode': 'exp',
        'coef': {
            'decay': 0.995
        }
    },
    'importance_retrieval': DEFAULT_VALUE_RETRIEVAL,
    'importance_judge': DEFAULT_IMPORTANCE_JUDGE,
    'topk': DEFAULT_RETRIEVAL_TOP_K,  
    'empty_memory': 'None'
}


DEFAULT_MGMEMORY_RECALL = {
    'method': 'MGMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'warning_threshold': 0.7,
    'warning_content': 'We suggest to execute memory_archive or memory_transfer.',
    'trigger': DEFAULT_MGMEMORY_LLMTRIGGER,
    'recall_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'archival_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_RFMEMORY_RECALL = DEFAULT_FUMEMORY_RECALL

# ----- Store -----
DEFAULT_FUMEMORY_STORE = {
    'method': 'FUMemoryStore'
}

DEFAULT_STMEMORY_STORE = {
    'method': 'LTMemoryStore'
}

DEFAULT_LTMEMORY_STORE = {
    'method': 'LTMemoryStore'
}


DEFAULT_GAMEMORY_STORE = {
    'method': 'GAMemoryStore'
}


DEFAULT_MGMEMORY_STORE = {
    'method': 'MGMemoryStore',
    'flush_checker': DEFAULT_TRUNCATION,
    'summarizer': {
        'method': 'LLMSummarizer',
        'LLM_config': DEFAULT_LLM,
        'prompt': {
            'template': """Recursive Summary: {recursive_summary}
Recent Memory: {flush_context}
Please update the Recursive Summary based on Recursive Summary and summarizing Recent Memory.
Just output the new Recursive Summary, without any other messages.""",
            'input_variables': ['recursive_summary', 'flush_context']
        }
    }
}


# ----- Reflect -----
DEFAULT_GAMEMORY_REFLECT = {
    'method': 'GAReflect',
    'reflector': DEFAULT_GAREFLECTOR
}

# ----- Optimize -----
DEFAULT_RFMEMORY_OPTIMIZE = {
    'method': 'RFMemoryOptimize',
    'reflector': DEFAULT_TRIALREFLECTOR
}