from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import MMMemoryRecall
from memengine.operation.Store import MMMemoryStore
from memengine.function.MultiModalRetrieval import MultiModalRetrieval
from memengine.utils.Display import *

class MMMemory(ExplicitMemory):
    """
    MMMemory (MultiModal Memory): 
    Supports text, image, and text+image storage and retrieval using multimodal embeddings.

    Based on the retrieval format in paper: 
    MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text
    https://aclanthology.org/2022.emnlp-main.375.pdf
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.storage = LinearStorage(self.config.args.storage)
        
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        self.store_op = MMMemoryStore(
            self.config.args.store, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        self.recall_op = MMMemoryRecall(
            self.config.args.recall, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )

        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display, 
            register_dict={'Memory Storage': self.storage}
        )
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        """
        Store observation (text, image, or both)
        """
        self.store_op(observation)
    
    def recall(self, query) -> str:
        """
        Recall relevant memories based on query
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass
