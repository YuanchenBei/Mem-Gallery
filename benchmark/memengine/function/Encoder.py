from abc import ABC, abstractmethod
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

class BaseEncoder(ABC):
    """
    Transfer textual messages into embeddings to represent in latent space by pre-trained models.
    """
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def reset(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class LMEncoder(BaseEncoder):
    """
    Embedding vias LM transformers.
    """
    def __init__(self, config):
        super().__init__(config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.path)
        self.model = AutoModel.from_pretrained(self.config.path).to(self.device)
        self.model.eval()
    
    def __call__(self, text, return_type = 'numpy'):
        res = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**res).last_hidden_state[:, -1, :]
        if return_type == 'numpy':
            return embeddings.numpy()
        elif return_type == 'tensor':
            return embeddings.to(self.device)
        else:
            return 'Unrecognized Return Type.'

class STEncoder(BaseEncoder):
    """
    Embedding vias Sentence Transformer.
    """
    def __init__(self, config):
        super().__init__(config)

        self.model = SentenceTransformer(self.config.path).to(self.device)
    
    def __call__(self, text, return_type = 'numpy'):
        embeddings = self.model.encode([text], normalize_embeddings=True)
        if return_type == 'numpy':
            return embeddings.cpu().numpy()
        elif return_type == 'tensor':
            return torch.from_numpy(embeddings).to(self.device)
        else:
            return 'Unrecognized Return Type.'