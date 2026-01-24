from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import GraphStorage
from memengine.operation.Recall import NGMemoryRecall
from memengine.operation.Store import NGMemoryStore
from memengine.function.MultiModalRetrieval import MultiModalRetrieval
from memengine.utils.Display import *

class NGMemory(ExplicitMemory):
    """
    NGMemory (Neural Graph Memory): 
    
    Reference: Neural Graph Memory: A Structured Approach to Long-Term Memory in Multimodal Agents
    https://www.researchgate.net/profile/Matt-Fisher-7/publication/394440420_Neural_Graph_Memory_A_Structured_Approach_to_Long-Term_Memory_in_Multimodal_Agents/links/689ab8c337b271210509c20f/Neural-Graph-Memory-A-Structured-Approach-to-Long-Term-Memory-in-Multimodal-Agents.pdf
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Use GraphStorage for graph-structured memory
        self.storage = GraphStorage(self.config.args.storage)
        
        # Create multimodal retrieval
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        # Create store and recall operations
        self.store_op = NGMemoryStore(
            self.config.args.store, 
            storage=self.storage,
            multimodal_retrieval=self.multimodal_retrieval
        )
        
        self.recall_op = NGMemoryRecall(
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
        Store observation (text, image, or both) as a graph node
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using query-aware graph traversal
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.node_counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

