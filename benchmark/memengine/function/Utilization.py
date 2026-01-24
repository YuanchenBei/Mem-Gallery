from abc import ABC, abstractmethod

class BaseUtilization(ABC):
    """
    Deal with several different parts of memory contents, formulating these information into a unified output.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ConcateUtilization(BaseUtilization):
    """
    Concate memory pieces into a string for application.
    """
    def __init__(self, config):
        super().__init__(config)

    def concate_list(self, l):
        # Process list, supports both string and dict formats
        processed_list = []
        for item in l:
            if isinstance(item, dict):
                # If dict, extract text and timestamp fields
                text = item.get('text', '')
                timestamp = item.get('timestamp') or item.get('time', '')

                # Combine text and timestamp
                if timestamp:
                    formatted_item = f"timestamp: {timestamp}\n{text}"
                else:
                    formatted_item = text

                processed_list.append(formatted_item)
            else:
                # If string or other type, use directly
                processed_list.append(str(item))

        # Apply index format (if configured)
        if self.config.list_config.index:
            processed_list = ['[Memory %d] %s' % (index, m) for index, m in enumerate(processed_list)]
        
        if len(processed_list) == 0:
            return 'None'
        
        return self.config.list_config.sep.join(processed_list)

    def __call__(self, input_memory):
        if isinstance(input_memory, list):
            main_body = self.concate_list(input_memory)
        elif isinstance(input_memory, dict):
            main_body_list = []
            for k,v in input_memory.items():
                if isinstance(v, list):
                    v = self.concate_list(v)
                
                main_body_list.append(self.config.dict_config.key_value_sep.join([self.config.dict_config.key_format % k, v]))
            main_body = self.config.dict_config.item_sep.join(main_body_list)
        elif isinstance(input_memory, str):
            main_body = input_memory
        
        return '\n'.join([self.config.prefix, main_body, self.config.suffix])


class MultiModalUtilization(BaseUtilization):
    """
    Formatting tool specifically for multimodal memory
    Handles text and images completely separately
    - Text part: only contains memories with actual text content
    - Image part: returns list of all image paths
    """
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, input_memory):
        """
        Process multimodal memory list

        Args:
            input_memory: list of dict, each dict contains:
                {
                    "text": "msg",
                    "image": {"path": "xxx",
                            "caption": "xxx",
                            "img_id": "xxx"},
                    "timestamp": xxx,
                    "dialogue_id": xxx,
                    "counter": xxx
                }

        Returns:
            dict: {
                'text': formatted_text,
                'image': image_list,
                'timestamp': timestamp_list,
                'dialogue_id': dialogue_id_list
            }
        """
        if not isinstance(input_memory, list):
            input_memory = [input_memory] if input_memory else []

        text_index = 0  
        output_dict_list = []
        
        for item in input_memory:
            if isinstance(item, dict):
                text = item.get('text', '').strip()
                image = item.get('image')
                timestamp = item.get('timestamp')
                dialogue_id = item.get('dialogue_id')
                
                if self.config.list_config.index:
                    text_parts = f'[Memory {text_index}] timestamp: {timestamp}\n{text}'
                    text_index += 1
                else:
                    text_parts = f'timestamp: {timestamp}\n{text}'
                
                output_dict_list.append({'text': text_parts, 'image': image, 'timestamp': timestamp, 'dialogue_id': dialogue_id})

        return output_dict_list
