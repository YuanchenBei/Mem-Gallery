from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import FUMemoryRecall
from memengine.operation.Store import FUMemoryStore
from memengine.utils.Display import *

class FUMemory(ExplicitMemory):
    """
    FUMemory (Full Memory): Naively concatenate all the information into one string, also known as long-context memory.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.store_op = FUMemoryStore(self.config.args.store, storage = self.storage)
        self.recall_op = FUMemoryRecall(self.config.args.recall, storage = self.storage)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage
        })
    
    def reset(self) -> None:
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