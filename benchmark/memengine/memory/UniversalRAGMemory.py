from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.UniversalRAGStorage import UniversalRAGStorage
from memengine.operation.Recall import UniversalRAGRecall
from memengine.operation.Store import UniversalRAGStore
from memengine.function.UniversalRAGRouting import UniversalRAGRouting
from memengine.function.UniversalRAGRetrieval import UniversalRAGRetrieval
from memengine.utils.Display import *

class UniversalRAGMemory(ExplicitMemory):
    """
    UniversalRAG Memory
    
    Features:
    - Dynamic routing: Automatically routes queries to 'no', 'document', or 'image'
    - Multi-modal retrieval: Supports text and image retrieval using GME embeddings
    - Unified interface: Compatible with existing memory framework

    Reference: UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
    https://arxiv.org/pdf/2504.20734
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = UniversalRAGStorage(self.config.args.storage)
        
        self.routing = UniversalRAGRouting(self.config.args.routing)
        
        self.retrieval = UniversalRAGRetrieval(
            self.config.args.retrieval,
            storage=self.storage
        )
        
        self.store_op = UniversalRAGStore(
            self.config.args.store,
            storage=self.storage
        )
        
        self.recall_op = UniversalRAGRecall(
            self.config.args.recall,
            storage=self.storage,
            routing=self.routing,
            retrieval=self.retrieval
        )
        
        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display,
            register_dict={'Memory Storage': self.storage}
        )
    
    def reset(self) -> None:
        self.__reset_objects__([
            self.storage, self.store_op, self.recall_op,
            self.routing, self.retrieval
        ])
    
    def store(self, observation) -> None:
        """
        Store observation (text, image, or both) with automatic feature extraction.
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using dynamic routing and multi-modal retrieval.
        Routes query to appropriate modality ('no', 'document', or 'image') and retrieves accordingly.
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
    
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass


















