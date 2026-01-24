from abc import ABC, abstractmethod
from memengine.function.LLM import *
from langchain.prompts import PromptTemplate
import ast
import re

class BaseTrigger(ABC):
    """
    Designed to call functions or tools in extensible manners.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LLMTrigger(BaseTrigger):
    """
    Utilizing LLMs to determine which function should be called with specific arguments. 
    """
    def __init__(self, config):
        super().__init__(config)

        self.llm = eval(config.LLM_config.method)(config.LLM_config)
        self.func_dict = {}

        for func in config.func_list:
            self.register(func)

    def register(self, func):
        new_func = {
            'name': func['name'],
            'args': func['args'],
            'args_type': func['args_type'],
            'func_description': func['func_description'],
            'args_description': func['args_description']
        }
        self.func_dict[func['name']] = new_func

    def __get_function_prompt__(self):
        function_content = '\n\n'.join(["""def %s(%s):
    Description: %s
    Args: %s""" % (func['name'], ','.join(func['args']), func['func_description'], ';'.join(func['args_description']))
        for func_name, func in self.func_dict.items()])
        return """----- Function Descriptions Start -----
%s
----- Function Descriptions End -----""" % function_content

    def __parse_excuate_function__(self, res):
        if hasattr(self.config, 'no_execuate'):
            if res.strip() == self.config.no_execuate:
                return False
        excuate_list = [q for q in res.splitlines() if q.strip() != '']
        return_list = []
        for exe_text in excuate_list:
            pattern = r'(\w+)\((.*?)\)'
            match = re.search(pattern, exe_text)
            if match:
                func_name, func_args = match.group(1), match.group(2)
                if func_name not in self.func_dict:
                    print('Execuate Parse Fail.')
                    continue

                func_meta = self.func_dict[func_name]
                expected_types = func_meta['args_type']
                raw_args_text = func_args.strip()

                parsed_raw_args = []
                if raw_args_text == '':
                    parsed_raw_args = []
                else:
                    try:
                        candidate = ast.literal_eval(f'[{raw_args_text}]')
                        if isinstance(candidate, (list, tuple)):
                            parsed_raw_args = list(candidate)
                        else:
                            parsed_raw_args = [candidate]
                    except (SyntaxError, ValueError):
                        parsed_raw_args = None

                if parsed_raw_args is None:
                    if len(expected_types) == 1 and expected_types[0] == 'str':
                        arg_text = raw_args_text
                        if (arg_text.startswith("'") and arg_text.endswith("'")) or (arg_text.startswith('"') and arg_text.endswith('"')):
                            arg_text = arg_text[1:-1]
                        parsed_raw_args = [arg_text]
                    else:
                        print('Execuate Parse Fail.')
                        continue

                if len(parsed_raw_args) != len(expected_types):
                    print('Execuate Parse Fail.')
                    continue

                parsed_args = []
                for ind, tp in enumerate(expected_types):
                    value = parsed_raw_args[ind]
                    if tp == 'str':
                        parsed_args.append(str(value))
                    elif tp == 'list':
                        if not isinstance(value, list):
                            print('Execuate Parse Fail.')
                            parsed_args = None
                            break
                        parsed_args.append(value)
                    else:
                        parsed_args.append(value)
                    

                if parsed_args is None:
                    continue

                return_list.append((func_name, parsed_args))
            else:
                print('Execuate Parse Fail.')
        return return_list

    def __call__(self, input_dict):
        input_dict['function_prompt'] = self.__get_function_prompt__()
        prompt_template = PromptTemplate(
            input_variables=self.config.prompt.input_variables,
            template=self.config.prompt.template
        )
        prompt = prompt_template.format(**input_dict)
        res = self.llm.fast_run(prompt)

        return self.__parse_excuate_function__(res)

