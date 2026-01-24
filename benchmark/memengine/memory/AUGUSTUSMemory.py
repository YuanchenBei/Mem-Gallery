from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import TagGraphStorage
from memengine.operation.Recall import AUGUSTUSMemoryRecall
from memengine.operation.Store import AUGUSTUSMemoryStore
from memengine.function.MultiModalRetrieval import MultiModalRetrieval
from memengine.function.ConceptExtractor import LLMConceptExtractor, SimpleConceptExtractor
from memengine.function.ConceptBasedRetrieval import ConceptBasedRetrieval
from memengine.utils.Display import *

class AUGUSTUSMemory(ExplicitMemory):
    """
    A multimodal memory system with concept-based retrieval and contextualized graph memory.
        
    Reference: AUGUSTUS: An LLM-Driven Multimodal Agent System with Contextualized User Memory
    https://arxiv.org/pdf/2510.15261
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        
        # Contextual Memory: Graph-structured memory with concept tags
        # As per AUGUSTUS paper: arangoDB for contextual memory
        # Using TagGraphStorage as an in-memory equivalent
        self.contextual_memory = TagGraphStorage(self.config.args.storage)
        
        # Create multimodal retrieval
        self.multimodal_retrieval = MultiModalRetrieval(self.config.args.multimodal_retrieval)
        
        # Create concept extractor
        extractor_method = getattr(self.config.args.concept_extractor, 'method', 'SimpleConceptExtractor')
        if extractor_method == 'LLMConceptExtractor':
            self.concept_extractor = LLMConceptExtractor(self.config.args.concept_extractor)
        elif extractor_method == 'SimpleConceptExtractor':
            self.concept_extractor = SimpleConceptExtractor(self.config.args.concept_extractor)
        else:
            raise ValueError(f"Unsupported concept extractor method: {extractor_method}")
        
        # Create concept-based retrieval
        self.concept_retrieval = ConceptBasedRetrieval(self.config.args.concept_retrieval)
        self.concept_retrieval.set_storage(self.contextual_memory)
        
        # Create store and recall operations
        self.store_op = AUGUSTUSMemoryStore(
            self.config.args.store, 
            contextual_memory=self.contextual_memory,
            multimodal_retrieval=self.multimodal_retrieval,
            concept_extractor=self.concept_extractor
        )
        
        self.recall_op = AUGUSTUSMemoryRecall(
            self.config.args.recall, 
            contextual_memory=self.contextual_memory,
            multimodal_retrieval=self.multimodal_retrieval,
            concept_extractor=self.concept_extractor,
            concept_retrieval=self.concept_retrieval
        )

        self.auto_display = eval(self.config.args.display.method)(
            self.config.args.display, 
            register_dict={
                'Contextual Memory': self.contextual_memory
            }
        )
    
    def reset(self) -> None:
        self.__reset_objects__([
            self.contextual_memory, 
            self.store_op, 
            self.recall_op, 
            self.concept_extractor, 
            self.concept_retrieval
        ])

    def store(self, observation) -> None:
        """
        Store observation in Contextual Memory (concept-tagged graph).
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall relevant memories using CoPe search from Contextual Memory.
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.contextual_memory.node_counter)
    
    def get_concept_statistics(self) -> dict:
        """
        Get statistics about concepts in the contextual memory graph.
        
        Returns:
            dict: Statistics including total concepts, most common concepts, etc.
        """
        return self.contextual_memory.get_concept_statistics()
    
    @property
    def storage(self):
        """
        Backward compatibility: return contextual_memory as storage.
        This allows existing code that references self.storage to continue working.
        """
        return self.contextual_memory

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

