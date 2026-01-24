from abc import ABC, abstractmethod
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate

class BaseReflector(ABC):
    """
    Draw new insights in higher level from existing information, commonly for reflection and optimization operations.
    """
    def __init__(self, config):
        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def generate_insight(self, *args, **kwargs):
        pass

class GAReflector(BaseReflector):
    def __init__(self, config):
        super().__init__(config)

        self.accmulated_importance = 0
        self.recursion_list = []
        self.llm = eval(config.LLM_config.method)(config.LLM_config)

    def reset(self):
        self.accmulated_importance = 0
        self.recursion_list = []
    
    def add_reflection(self, importance, recursion):
        self.accmulated_importance += importance
        self.recursion_list.append(recursion)
    
    def delete_reflection(self, index):
        raise NotImplementedError()
    
    def get_recursion(self, index):
        return self.recursion_list[index]

    def get_current_accmulated_importance(self):
        return self.accmulated_importance
    
    def get_reflection_threshold(self):
        return self.config.threshold

    def self_ask(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.question_prompt.input_variables,
            template=self.config.question_prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        question_list = [q for q in res.splitlines() if q.strip() != '']
        assert len(question_list) == self.config.question_number

        return question_list
    
    def generate_insight(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.insight_prompt.input_variables,
            template=self.config.insight_prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        insight_list = [q for q in res.splitlines() if q.strip() != '']

        assert len(insight_list) ==  self.config.insight_number

        return insight_list

class TrialReflector(BaseReflector):
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)
    
    def generate_insight(self, input_dict):
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        return res