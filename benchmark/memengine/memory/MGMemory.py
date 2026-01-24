from memengine.memory.BaseMemory import ExplicitMemory
from memengine.utils.Storage import LinearStorage
from memengine.operation.Recall import MGMemoryRecall
from memengine.operation.Store import MGMemoryStore
from memengine.utils.Display import *

class MGMemory(ExplicitMemory):
    """
    MGMemory (MemGPT [1]): A hierarchical memory model that treat the memory system as an operation system.

    [1] Packer, Charles, et al. "Memgpt: Towards llms as operating systems." arXiv preprint arXiv:2310.08560 (2023).
    https://arxiv.org/pdf/2310.08560
    """
    def __init__(self, config):
        super().__init__(config)

        self.main_context = {
            'working_context': LinearStorage(self.config.args.storage),
            'FIFO_queue': LinearStorage(self.config.args.storage),
            'recursive_summary': {'global': 'None'}
        }
        self.recall_storage = LinearStorage(self.config.args.storage)
        self.archival_storage = LinearStorage(self.config.args.storage)

        self.recall_op = MGMemoryRecall(
            self.config.args.recall,
            main_context = self.main_context,
            recall_storage = self.recall_storage,
            archival_storage = self.archival_storage
        )
        self.store_op = MGMemoryStore(
            self.config.args.store,
            main_context = self.main_context,
            recall_storage = self.recall_storage,
            recall_retrieval = self.recall_op.recall_retrieval,
            truncation = self.recall_op.truncation
        )

        self.auto_display = eval(self.config.args.display.method)(self.config.args.display, register_dict = {
            'Working Memory': self.main_context['working_context'],
            'Recursive Memory Summary': self.main_context['recursive_summary'],
            'FIFO Memory': self.main_context['FIFO_queue'],
            'Recall Storage': self.recall_storage,
            'Archival Storage': self.archival_storage
        })
        
    def reset(self) -> None:
        self.main_context = {
            'working_context': LinearStorage(self.config.args.storage),
            'FIFO_queue': LinearStorage(self.config.args.storage),
            'recursive_summary': {'global': 'None'}
        }
        self.__reset_objects__([self.recall_storage, self.archival_storage, self.recall_op, self.store_op])
    
    def store(self, observation) -> None:
        self.store_op(observation)

    def recall(self, observation) -> object:
        return self.recall_op(observation)
    
    def display(self) -> None:
        self.auto_display('-')

    def manage(self) -> None:
        pass
    
    def optimize(self, **kwargs) -> None:
        pass