from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import RFMemoryRecall
from memengine.operation.Store import FUMemoryStore
from memengine.operation.Optimize import RFOptimize
from memengine.utils.Display import *

class RFMemory(ExplicitMemory):
    """
    RFMemory (Reflexion [1]): A famous memory method that can learn to memorize from previous trajectories by optimization.

    [1] Shinn, Noah, et al. "Reflexion: Language agents with verbal reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).
    https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf
    """
    def __init__(self, config):
        super().__init__(config)

        self.storage = LinearStorage(self.config.args.storage)
        self.insight = {'global_insight': ''}

        self.store_op = FUMemoryStore(self.config.args.store, storage = self.storage)
        self.recall_op = RFMemoryRecall(self.config.args.recall, storage = self.storage, insight = self.insight)
        self.optimize_op = RFOptimize(self.config.args.optimize, insight = self.insight)

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Memory Storage': self.storage,
            'Insight': self.insight
        })

    def reset(self):
        self.insight = {'global_insight': ''}
        self.__reset_objects__([self.storage, self.store_op, self.recall_op, self.optimize_op])

    def store(self, observation) -> None:
        self.store_op(observation)
    
    def recall(self, query) -> object:
        return self.recall_op(query)
    
    def display(self) -> None:
        self.auto_display(self.storage.counter)

    def manage(self, operation, **kwargs) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        self.optimize_op(**kwargs)


