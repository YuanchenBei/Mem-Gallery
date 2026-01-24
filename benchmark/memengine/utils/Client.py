import requests, os

def __store_convert_str_to_observation__(method):
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper

class Client():
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_id = self.get_session_id()

    def request(self, route_path, data):
        post_path = os.path.join(self.base_url,route_path)
        response = requests.post(post_path, json=data)

        if response.status_code == 200:
            return response.json()
        else:
            return False
    
    def get_session_id(self):
        response = self.request('init/', data=None)
        if response:
            print('Successully start client-server service.')
            return response['session_id']
        else:
            raise "Initializing client fails."
    
    def initilize_memory(self, method, config_dict):
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'initilize',
            'kwargs': {
                'method': method,
                'config': config_dict
            }
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']

    def reset(self) -> None:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'reset',
            'kwargs': {}
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']

    @__store_convert_str_to_observation__
    def store(self, obervation) -> None:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'store',
            'kwargs': obervation
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']
    
    @__store_convert_str_to_observation__
    def recall(self, query) -> object:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'recall',
            'kwargs': query
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']
    
    def manage(self, operation, **kwargs) -> None:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'manage',
            'kwargs': {
                'operation': operation,
                'kwargs': kwargs
            }
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']

    def display(self) -> None:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'display',
            'kwargs': {}
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']
    
    def optimize(self, **kwargs) -> None:
        response = self.request('operation/', data={
            'session_id': self.session_id,
            'operation': 'optimize',
            'kwargs': kwargs
        })
        if response:
            print(response['info'])
            if 'response' in response:
                return response['response']