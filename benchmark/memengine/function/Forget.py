from abc import ABC, abstractmethod

import math, random

class BaseForget(ABC):
    """
    Typically applied in simulation-oriented agents, such role-playing and social simulations.
    It empowers agents with features of human cognitive psychology, aligning with human roles.
    """
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def get_forget_prob(self, *args, **kwargs):
        pass

class MBForget(BaseForget):
    """
    Forgetting according to an exponential curve.
    According to the paper: Memorybank: Enhancing large language models with long-term memory
    https://ojs.aaai.org/index.php/AAAI/article/download/29946/31654
    """
    def __init__(self, config):
        super().__init__(config)

    def get_forget_prob(self, current_time, recency, strength):
        return math.exp(-(current_time-recency)/self.config.coef*strength)
    
    def sample_forget(self, current_time, recency, strength):
        return random.random() > self.get_forget_prob(current_time, recency, strength)