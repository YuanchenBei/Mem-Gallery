from abc import ABC, abstractmethod
import os

class BaseDisplay(ABC):
    def __init__(self, config, register_dict) -> None:
        super().__init__()

        self.config = config
        self.register_dict = register_dict

    def reset(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

class TextDisplay(BaseDisplay):
    """
    Display memory contents in text forms.
    """
    def __init__(self, config, register_dict) -> None:
        super().__init__(config, register_dict)
    
    def __get_one_item__(self, name, obj):
        if isinstance(obj, dict):
            return self.config.key_value_sep.join([self.config.key_format % name, list(obj.values())[0]])
        elif hasattr(obj, 'display') and callable(getattr(obj, 'display')):
            return self.config.key_value_sep.join([self.config.key_format % name, obj.display()])

    def __get_display_memory__(self, tag):
        current_memory = self.config.item_sep.join([self.__get_one_item__(k, v) for k, v in self.register_dict.items()])
        return '\n'.join([self.config.prefix % str(tag), current_memory, self.config.suffix])

class ScreenDisplay(TextDisplay):
    """
    Display memory contents in the console.
    """
    def __init__(self, config, register_dict):
        super().__init__(config, register_dict)

    def __call__(self, tag):
        print(self.__get_display_memory__(tag))

class FileDisplay(TextDisplay):
    """
    Display memory contents in the file.
    """
    def __init__(self, config, register_dict) -> None:
        super().__init__(config, register_dict)

        dir_path = os.path.dirname(self.config.output_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(self.config.output_path, 'w', encoding='utf-8') as file:
            pass
    
    def __call__(self, tag):
        with open(self.config.output_path, 'a', encoding='utf-8') as file:
            file.write(self.__get_display_memory__(tag))