from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import GAMemoryRecall
from memengine.operation.Store import GAMemoryStore
from memengine.operation.Reflect import GAReflect
from memengine.utils.Display import *

class GAMemory(ExplicitMemory):
    """
    GAMemory (Generative Agents [1]): A pioneer memory model with weighted retrieval combination and self-reflection mechanism.
    
    [1] Park, Joon Sung, et al. "Generative agents: Interactive simulacra of human behavior." Proceedings of the 36th annual acm symposium on user interface software and technology. 2023.
    https://dl.acm.org/doi/pdf/10.1145/3586183.3606763
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = GAMemoryRecall(self.config.args.recall, storage = self.storage)
        self.reflect_op = GAReflect(
            self.config.args.reflect,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval
        )
        self.store_op = GAMemoryStore(
            self.config.args.store,
            storage = self.storage,
            text_retrieval = self.recall_op.text_retrieval,
            time_retrieval = self.recall_op.time_retrieval,
            importance_retrieval = self.recall_op.importance_retrieval,
            imporatance_judge = self.recall_op.imporatance_judge,
            reflector = self.reflect_op.reflector
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.reflect_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
        
    def manage(self, operation, **kwargs) -> None:
        if operation == 'reflect':
            insight_list = self.reflect_op(**kwargs)
            for insight in insight_list:
                self.store_op(insight)
        else:
            raise "Management error."
    
    def optimize(self, **kwargs) -> None:
        pass