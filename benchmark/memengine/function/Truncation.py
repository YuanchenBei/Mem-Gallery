from abc import ABC, abstractmethod
from transformers import AutoTokenizer

class BaseTruncation(ABC):
    """
    Helps to formulate memory contexts under the limitations of token number by certain LLMs.
    """
    def __init__(self, config) -> None:
        self.config = config
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMTruncation(BaseTruncation):
    def __init__(self, config):
        super().__init__(config)

        if self.config.mode == 'token':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)

    def truncate_by_word(self, text):
        words = text.split(' ')
        if len(words) > self.config.number:
            words = words[-self.config.number:]
        return ' '.join(words)

    def truncate_by_token(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.config.number:
            tokens = tokens[-self.config.number:]
        truncated_text = self.tokenizer.convert_tokens_to_string(tokens)
        return truncated_text

    def get_piece_number(self, text):
        if self.config.mode == 'word':
            return len(text.split(' '))
        elif self.config.mode == 'token':
            return len(self.tokenizer.tokenize(text))
        else:
            raise "Truncation mode error."

    def check_truncation_needed(self, text):
        return self.get_piece_number(text) > self.config.number

    def __call__(self, text):
        if self.config.mode == 'word':
            return self.truncate_by_word(text)
        elif self.config.mode == 'token':
            return self.truncate_by_token(text)
        else:
            raise "Truncation mode error."


class MMLMTruncation(BaseTruncation):
    """
    MultiModal Language Model Truncation:
    Truncates multimodal memory based on token budget (text tokens + image tokens).
    Formula: N_text + (N_image × T_img) ≤ L_max
    
    Truncates from the oldest memories (keeps the newest ones).
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize tokenizer for text token counting
        if self.config.mode == 'token':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
        else:
            self.tokenizer = None
        
        # Tokens per image (T_img)
        # Default: 256 for Qwen series, 576 for LLaVA/GPT/Gemini series
        self.tokens_per_image = getattr(self.config, 'tokens_per_image', 256)
        
        # Maximum total tokens (L_max)
        self.max_tokens = self.config.number
    
    def reset(self):
        pass
    
    def _count_text_tokens(self, text):
        """Count tokens in text."""
        if self.config.mode == 'token' and self.tokenizer:
            return len(self.tokenizer.tokenize(text))
        elif self.config.mode == 'word':
            return len(text.split(' '))
        else:
            raise "Truncation mode error."
    
    def _count_image_tokens(self, memory_item):
        """Count tokens for images in a memory item."""
        image = memory_item.get('image')
        if image and image is not None:
            # Check if image is a dict with path/caption, or just a path string
            if isinstance(image, dict):
                # Has image
                return self.tokens_per_image
            elif isinstance(image, str) and image.strip():
                # Image path string (non-empty)
                return self.tokens_per_image
        return 0
    
    def _count_memory_tokens(self, memory_item):
        """Count total tokens for a memory item (text + images)."""
        text = memory_item.get('text', '')
        text_tokens = self._count_text_tokens(text)
        image_tokens = self._count_image_tokens(memory_item)
        return text_tokens + image_tokens
    
    def __call__(self, memory_list):
        """
        Truncate multimodal memory list based on token budget AND image count limit.
        
        Logic:
        - If memory has image: check both image count limit AND token budget
        - If memory has no image: only check token budget
        - This allows adding text-only memories even after image limit is reached
        
        Args:
            memory_list: list of dict, each dict is a memory item with:
                - 'text': str
                - 'image': dict or None
                - 'timestamp': int/float
                - 'dialogue_id': int
                - etc.
        
        Returns:
            list: Truncated memory list (keeping newest memories)
        """
        if not memory_list:
            return []
        
        # Maximum number of images allowed (default: 5, leaving 1 for query image)
        max_images = getattr(self.config, 'max_images', 5)
        
        # Process from newest to oldest (reverse order)
        result = []
        total_tokens = 0
        total_images = 0
        
        # Iterate from newest (last) to oldest (first)
        for mem in reversed(memory_list):
            mem_tokens = self._count_memory_tokens(mem)
            has_image = self._count_image_tokens(mem) > 0
            
            # Check if memory can be added
            can_add = False
            
            if has_image:
                # Memory has image: check both image count limit AND token budget
                if (total_images < max_images and 
                    total_tokens + mem_tokens <= self.max_tokens):
                    can_add = True
            else:
                # Memory has no image: only check token budget
                if total_tokens + mem_tokens <= self.max_tokens:
                    can_add = True
            
            if can_add:
                # Insert at beginning to maintain chronological order (oldest first)
                result.insert(0, mem)
                total_tokens += mem_tokens
                if has_image:
                    total_images += 1
            else:
                # Cannot add this memory
                # If it has image and we're at image limit, continue to check next (might be text-only)
                # If token budget exceeded, stop (no more memories can fit)
                if not has_image and total_tokens + mem_tokens > self.max_tokens:
                    # Token budget exceeded, stop adding more (older) memories
                    break
                # Otherwise, continue to next memory (might be text-only that can fit)
                continue
        
        return result