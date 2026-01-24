from abc import ABC, abstractmethod

from openai import OpenAI

class BaseLLM(ABC):
    """
    Provides a convenient interface to utilize the powerful capability of different large language models.
    """
    def __init__(self, config):
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def fast_run(self, query):
        pass

class APILLM(BaseLLM):
    """
    Utilize LLM from APIs.
    """
    def __init__(self, config):
        super().__init__(config)

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content,
            'input_token': response.usage.prompt_tokens,
            'output_token': response.usage.completion_tokens
        }

    def run(self, message_list):
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=message_list,
            temperature=self.config.temperature
        )
        response = self.parse_response(response)
        return response

    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response['result']

class LocalVLLM(BaseLLM):
    def __init__(self, config):
        super().__init__(config)

        from vllm import LLM, SamplingParams
        self.model = LLM(config.name)

        self.sampling_params = SamplingParams(temperature=config.temperature)
    
    def run(self, message_list):
        return self.model.chat(message_list,self.sampling_params)[0]
    
    def fast_run(self, query):
        response = self.run([{"role": "user", "content": query}])
        return response.outputs[0].text
