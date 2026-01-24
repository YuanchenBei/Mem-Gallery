from default_config.DefaultOperationConfig import *
from default_config.DefaultUtilsConfig import *
from default_config.DefaultGlobalConfig import *
from default_config.DefaultMMMemoryConfig import *  
from default_config.DefaultMMFUMemoryConfig import *  
from default_config.DefaultNGMemoryConfig import *  
from default_config.DefaultAUGUSTUSMemoryConfig import *  
from default_config.DefaultUniversalRAGMemoryConfig import *  

DEFAULT_FUMEMORY = {
    'name': 'FUMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_STMEMORY = {
    'name': 'STMMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_STMEMORY_RECALL,
    'store': DEFAULT_STMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_LTMEMORY = {
    'name': 'LTMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_LTMEMORY_RECALL,
    'store': DEFAULT_LTMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_GAMEMORY = {
    'name': 'GAMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_GAMEMORY_RECALL,
    'store': DEFAULT_GAMEMORY_STORE,
    'reflect': DEFAULT_GAMEMORY_REFLECT,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}


DEFAULT_MGMEMORY = {
    'name': 'MGMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_MGMEMORY_RECALL,
    'store': DEFAULT_MGMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_RFMEMORY = {
    'name': 'RFMemory',
    'is_multimodal': False,
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_FUMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'optimize': DEFAULT_RFMEMORY_OPTIMIZE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_ALL_PARAM = DEFAULT_FUMEMORY