from copy import deepcopy
from memengine import *

def generate_candidate(adjust_dict):
    """Generate a list of configs with combination (Recommended for hyper-parameter tuning).

    Args:
        adjust_dict (dict): a dict that includes
            'model' (str): the name of memory model, such as 'LTMemory'
            'base_config' (str): the common basic config of the memory model, such as DEFAULT_LTMEMORY
            'adjust_name' (str): the parameter path in config (seerated by dot), such as 'recall.text_retrieval.topk'
            'adjust_range' (list): the list of values for parameter tuning [1, 3, 5, 10]

    Returns:
        list: a list of combinational configs
    """
    candidate_list = []
    for value in adjust_dict['adjust_range']:
        new_config = deepcopy(adjust_dict['base_config'])
        config_pointer = adjust_dict['adjust_name'].split('.')
        
        current_obj = new_config
        for p in config_pointer[:-1]:
            current_obj = current_obj[p]
        current_obj[config_pointer[-1]] = value

        candidate_list.append({
            'model': adjust_dict['model'],
            'config': new_config
        })
    return candidate_list

def automatic_select(reward_func, model_candidate):
    """Given the reward function, select which memory can perform best.

    Args:
        reward_func (function): the function whose input is memory and output is a float score.
        model_candidate (list): the list of model candidates.

    Returns:
        list: a sorted list of candidates with scores to show their performances.
    """
    result_list = []

    for mc in model_candidate:
        memory = eval(mc['model'])(MemoryConfig(mc['config']))
        reward = reward_func(memory)
        result_list.append(reward)
    
    sorted_with_index = sorted(enumerate(result_list), key=lambda x: x[1], reverse=True)
    sorted_result = [{
        'candidate_index': index,
        'score': value,
        'model': model_candidate[index]['model'],
        'config': model_candidate[index]['config']
    } for index, value in sorted_with_index]

    return sorted_result
