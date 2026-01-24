from abc import ABC, abstractmethod
from memengine.function.Encoder import *
from memengine.function.MultiModalEncoder import CLIPEncoder, GMEEncoder, MMEmbedEncoder
import numpy as np
import torch

class MultiModalRetrieval:
    """
    Multimodal retrieval: Supports text, image, and mixed queries.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        encoder_method = getattr(config.encoder, 'method', 'CLIPEncoder')
        if encoder_method == 'CLIPEncoder':
            self.encoder = CLIPEncoder(self.config.encoder)
        elif encoder_method == 'GMEEncoder':
            self.encoder = GMEEncoder(self.config.encoder)
        elif encoder_method == 'MMEmbedEncoder':
            self.encoder = MMEmbedEncoder(self.config.encoder)
        else:
            raise ValueError(f"Unsupported encoder method: {encoder_method}")
        
        # The tensorstore stores the embedding vectors of all memory.
        self.tensorstore = None
        
        # Storing metadata for each memory
        self.memory_metadata = []
    
    def reset(self):
        self.tensorstore = None
        self.memory_metadata = []
    
    def __normalize__(self, embedding):
        return torch.nn.functional.normalize(embedding, dim=-1)
    
    def add(self, obj):
        """
        Add an observation embedding to the retriever
        
        Args:
            obj: str或dict {'text': ..., 'image': ...}
        """
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        if self.tensorstore is None:
            self.tensorstore = embedding
        else:
            self.tensorstore = torch.cat([self.tensorstore, embedding], dim=0)
        
        metadata = {
            'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj and obj['text']),
            'has_image': isinstance(obj, dict) and 'image' in obj and obj['image']
        }
        self.memory_metadata.append(metadata)
        
        return embedding
    
    def __calculate_scores__(self, query):
        """
        Calculate the similarity score between the query and all stored memories.
        """
        query_embedding = self.encoder(query, return_type='tensor')
        
        if self.config.mode == 'cosine':
            query_embedding = self.__normalize__(query_embedding)
        
        if self.config.mode in ['cosine', 'dot']:
            scores = torch.matmul(self.tensorstore, query_embedding.squeeze())
        elif self.config.mode == 'L2':
            scores = - torch.norm(
                self.tensorstore - query_embedding.squeeze(), 
                p=2, 
                dim=1
            )
        else:
            raise ValueError(f"Unrecognized mode: {self.config.mode}")
        
        return scores
    
    def __call__(self, query, topk='config', with_score=False, sort=True):
        """
        Search for the most similar memories to the query.
        """
        if self.tensorstore is None or self.tensorstore.size(0) == 0:
            return torch.tensor([]) if not with_score else (torch.tensor([]), torch.tensor([]))
        
        scores = self.__calculate_scores__(query)
        
        if sort:
            scores, indices = torch.sort(scores, descending=True)
        else:
            indices = torch.arange(self.tensorstore.size(0))
        
        if topk is False:
            pass
        elif topk == 'config':
            k = min(self.config.topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        elif isinstance(topk, int):
            k = min(topk, self.tensorstore.size(0))
            scores = scores[:k]
            indices = indices[:k]
        
        if with_score:
            return scores, indices
        else:
            return indices
    
    def update(self, index, obj):
        """Update the embedding at a certain location"""
        embedding = self.encoder(obj, return_type='tensor')
        
        if self.config.mode == 'cosine':
            embedding = self.__normalize__(embedding)
        
        self.tensorstore[index] = embedding.squeeze()
        
        if index < len(self.memory_metadata):
            self.memory_metadata[index] = {
                'has_text': isinstance(obj, str) or (isinstance(obj, dict) and 'text' in obj and obj['text']),
                'has_image': isinstance(obj, dict) and 'image' in obj and obj['image']
            }
    
    def delete(self, index):
        """Delete the embedding at a specific location"""
        self.tensorstore = torch.cat([
            self.tensorstore[:index], 
            self.tensorstore[index+1:]
        ])
        
        if index < len(self.memory_metadata):
            self.memory_metadata.pop(index)
    
    def get_tensor_by_ids(self, id_list):
        return self.tensorstore[id_list]