from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import STMemoryRecall
from memengine.operation.Store import STMemoryStore
from memengine.utils.Display import *

class STMemory(ExplicitMemory):
    """
    STMemory (Short-term Memory, FIFO): Maintain the most recent information and concatenate them into one string as the context.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.recall_op = STMemoryRecall(self.config.args.recall, storage = self.storage)
        self.store_op = STMemoryStore(self.config.args.store, storage = self.storage, time_retrieval = self.recall_op.time_retrieval)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })

    def reset(self):
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)
    
    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass