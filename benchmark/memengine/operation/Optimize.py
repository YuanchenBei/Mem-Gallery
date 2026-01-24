from abc import ABC, abstractmethod
from memengine.function.Reflector import *

class BaseOptimize(ABC):
    def __init__(self, config):
        self.config = config
    
    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass

    @ abstractmethod
    def __call__(self, **kwargs):
        pass

class RFOptimize(BaseOptimize):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.reflector = TrialReflector(config.reflector)
        self.insight = kwargs['insight']
    
    def reset(self):
        self.__reset_objects__([self.reflector])

    def __call__(self, **kwargs):
        new_trial = kwargs['new_trial']

        new_insight = self.reflector.generate_insight({
            'previous_insight': self.insight['global_insight'],
            'new_trial': new_trial,
            'example': self.config.reflector.example
        })

        self.insight['global_insight'] = new_insight