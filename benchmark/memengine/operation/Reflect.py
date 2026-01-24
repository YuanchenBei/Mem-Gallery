from abc import ABC, abstractmethod
from memengine.function.Reflector import *

class BaseReflect(ABC):
    def __init__(self, config):
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

class GAReflect(BaseReflect):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.reflector = GAReflector(config.reflector)
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.time_retrieval = kwargs['time_retrieval']

    def reset(self):
        self.__reset_objects__([self.reflector])

    def __get_recursion_context__(self, mid):
        source = self.reflector.recursion_list[mid]
        text = self.storage.get_memory_text_by_mid(mid)
        if not source:
            return text
    
        return '%s [%s]' % (text, ';'.join([self.__get_recursion_context__(submid) for submid in source]))

    def __call__(self):
        if self.reflector.get_current_accmulated_importance() >= self.reflector.get_reflection_threshold():
            # Retrieve most recent information for reflection.
            ref_ids = self.time_retrieval(self.storage.counter, topk = self.config.reflector.reflection_topk)

            ref_context = '\n'.join([self.storage.get_memory_text_by_mid(mid) for mid in ref_ids])
            self.reflector.accmulated_importance = 0

            self.accmulated_importance = 0

            # Generate several questions with self-asking.
            question_list = self.reflector.self_ask({
                'information': ref_context,
                'question_number': self.config.reflector.question_number
            })

            ret_context = ''
            ret_evidence_list = []
            
            # Generate several insights for each question.
            for question in question_list:
                ret_ids = self.text_retrieval(question, topk = self.config.reflector.reflection_topk, sort = True)
                ret_evidence_list += ret_ids.cpu().numpy().tolist()

            ret_context = '\n'.join([self.__get_recursion_context__(mid) for mid in ret_evidence_list])
            
            # Generate several insights for each question.
            insight_list = self.reflector.generate_insight({
                'statements': ret_context,
                'insight_number': self.config.reflector.insight_number
            })
            
            # Generated insights need to include both timestamp and time fields (for consistency)
            current_counter = self.storage.counter
            return [{
                    'text': insight,
                    'time': current_counter,  # Numeric value, used for time_retrieval
                    'timestamp': f"Insight summary (counter: {current_counter})",  # Custom string, used for display
                    'source': ret_evidence_list
                }
                for insight in insight_list
            ]