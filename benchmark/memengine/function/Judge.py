from abc import ABC, abstractmethod
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate

class BaseJudge(ABC):
    """
    Assess given observations or intermediate messages on certain aspects.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LLMJudge(BaseJudge):
    """
    Judge vias large language models.
    """
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)

    def __post_scale__(self, res):
        # [TODO] Add an exception catch.
        # For example:
        # try:
        #     score = float(eval(res))
        # except Exception as e:
        #     score = 5.0
        score = float(eval(res))
        if hasattr(self.config, 'post_scale'):
            return score/self.config.post_scale
    
    def __post_bool__(self, res):
        if res == 'True':
            return True
        elif res == 'False':
            return False
        else:
            return "LLM Judge Parse Error for Boolean"

    def __call__(self, input_dict, post_process = 'scale'):
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        if post_process == 'scale':
            return self.__post_scale__(res)
        elif post_process == 'bool':
            return self.__post_bool__(res)
        else:
            raise "Judge Post Process Type Error!"
    