from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import MMFUMemoryRecall
from memengine.operation.Store import MMFUMemoryStore
from memengine.utils.Display import *

class MMFUMemory(ExplicitMemory):
    """
    MMFUMemory (MultiModal Full Memory): 
    Naively concatenate all multimodal information (text + images) and truncate based on token budget.
    
    Unlike FUMemory which converts images to captions, MMFUMemory preserves original image data.
    Truncation is based on: N_text + (N_image × T_img) ≤ L_max
    
    Key features:
    - Preserves both text and image data
    - Uses token budget-based truncation (text tokens + image tokens)
    - Keeps the newest memories when truncating
    - Uses MultiModalUtilization for formatting
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.store_op = MMFUMemoryStore(self.config.args.store, storage=self.storage)
        self.recall_op = MMFUMemoryRecall(self.config.args.recall, storage=self.storage)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict={
            'Memory Storage': self.storage
        })
    
    def reset(self) -> None:
        self.__reset_objects__([self.storage, self.store_op, self.recall_op])

    def store(self, observation) -> None:
        """
        Store observation (text, image, or both).
        The storage preserves both text and image data.
        """
        self.store_op(observation)
    
    def recall(self, query) -> object:
        """
        Recall all memories with token budget-based truncation.
        Returns multimodal data (text + images) instead of just text.
        """
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass

