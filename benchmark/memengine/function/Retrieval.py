from abc import ABC, abstractmethod

from memengine.function.Encoder import *
import numpy as np

class BaseRetrieval(ABC):
    """
    Utilized to find most useful information for the current query or observation.
    Commonly by different aspects like semantic relevance, importance, recency and so on.
    """
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def add(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def delete(self, **kwargs):
        pass

class LinearRetrieval(BaseRetrieval):
    """
    Retrieval in linear indexes.
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self):    
        self.tensorstore = None
    
    def __call__(self, query, topk = 'config', with_score = False, sort = True):
        scores = self.__calculate_scores__(query)
        
        if sort:
            scores, indices = torch.sort(scores, descending=True)
        else:
            indices = torch.arange(self.tensorstore.size(0))
        
        if topk is False:
            pass
        elif topk == 'config':
            scores, indices = scores[:self.config.topk], indices[:self.config.topk]
        elif isinstance(topk, int):
            scores, indices = scores[:topk], indices[:topk]
        
        if with_score:
            return scores, indices
        else:
            return indices

    def delete(self, index):
        self.tensorstore = torch.cat((self.tensorstore[:index], self.tensorstore[index+1:]))

    def get_tensor_by_ids(self, id_list):
        return self.tensorstore[id_list]

    @abstractmethod
    def __calculate_scores__(self, query):
        pass

class TextRetrieval(LinearRetrieval):
    """
    Retrieval most relevant texts accoring to the query.
    """
    def __init__(self, config):
        super().__init__(config)

        self.encoder = eval(self.config.encoder.method)(self.config.encoder)

    def __normalize__(self, embedding):
        return torch.nn.functional.normalize(embedding)

    def __calculate_scores__(self, query):
        query_embedding = self.encoder(query, return_type='tensor')
        if self.config.mode == 'cosine':
            query_embedding = self.__normalize__(query_embedding)

        if self.config.mode in ['cosine', 'dot']:
            scores = torch.matmul(self.tensorstore, query_embedding.squeeze())
        elif self.config.mode == 'L2':
            scores = - torch.norm(self.tensorstore - query_embedding.squeeze(), p=2, dim=1)
        else:
            raise "Unrecgonized faiss mode %s." % self.config.mode

        return scores

    def add(self, text):
        text_embedding = self.encoder(text, return_type='tensor')
        if self.config.mode == 'cosine':
            text_embedding = self.__normalize__(text_embedding)

        if self.tensorstore is None:
            self.tensorstore = text_embedding
        else:
            self.tensorstore = torch.vstack((self.tensorstore, text_embedding))
        
        return text_embedding
    
    def update(self, index, text):
        text_embedding = self.encoder(text, return_type='tensor')
        self.tensorstore = torch.cat((self.tensorstore[:index], text_embedding, self.tensorstore[index+1:]))

class ValueRetrieval(LinearRetrieval):
    """
    Retrieval certain values with several algorithms.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def __calculate_scores__(self, query):
        if self.config.mode == 'identical':
            scores = self.tensorstore
        elif self.config.mode == 'delta':
            scores = torch.tensor(query).to(self.device) - self.tensorstore
        elif self.config.mode == 'exp':
            delta = torch.tensor(query).to(self.device) - self.tensorstore
            scores = torch.pow(self.config.coef.decay, delta)
        else:
            raise "Value Retrieval mode error."
        
        return scores
    
    def add(self, value):
        if self.tensorstore is None:
            self.tensorstore = torch.tensor([]).to(self.device)
        self.tensorstore = torch.cat([self.tensorstore, torch.tensor([value]).to(self.device)])
    
    def update(self, index, value):
        self.tensorstore = torch.cat((self.tensorstore[:index], torch.tensor([value]).to(self.device), self.tensorstore[index+1:]))

class TimeRetrieval(ValueRetrieval):
    """
    Retrieval according to timestamps with several algorithms.
    """
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, query, topk = 'config', with_score = False, sort = True):
        if self.config.mode == 'raw':
            scores = torch.flip(self.tensorstore, dims=[0])
            indices = torch.arange(self.tensorstore.size(0)).flip(dims=[0])
        else:
            scores = self.__calculate_scores__(query)
            if sort:
                scores, indices = torch.sort(scores, descending=True)
            else:
                indices = torch.arange(self.tensorstore.size(0))
        
        if topk is False:
            pass
        elif topk == 'config':
            scores, indices = scores[:self.config.topk], indices[:self.config.topk]
        elif isinstance(topk, int):
            scores, indices = scores[:topk], indices[:topk]
        
        if with_score:
            return scores, indices
        else:
            return indices
