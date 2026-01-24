from memengine.utils.Storage import BaseStorage
from memengine.function.MultiModalEncoder import GMEEncoder
import torch

class UniversalRAGStorage(BaseStorage):
    """
    UniversalRAG Storage
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.memory_list = []
        self.counter = 0
        
        self._init_encoder(config)
        
        self.document_index = {}      
        self.image_index = {}          
    
    def _init_encoder(self, config):
        """Initialize encoder"""
        encoder_config = getattr(config, 'encoder', None)
        if encoder_config:
            self.encoder = GMEEncoder(encoder_config)
        else:
            from default_config.DefaultMMMemoryConfig import DEFAULT_GME_ENCODER
            self.encoder = GMEEncoder(DEFAULT_GME_ENCODER)
    
    def reset(self):
        self.memory_list = []
        self.counter = 0
        self.document_index = {}
        self.image_index = {}
    
    def is_empty(self):
        return self.counter == 0
    
    def display(self):
        memory_display_items = []
        for m in self.memory_list:
            memory_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in m.items()]))
        if len(memory_display_items) == 0:
            return 'None'
        return '\n'.join(['[Memory Entity %d]\n%s' % (index, m) for index, m in enumerate(memory_display_items)])
    
    def add(self, obj):
        """
        Add observations and extract features (using the GME encoder)
        - document_index: Extract text features (for text retrieval)
        - image_index: Extract image features (for image retrieval)
        """
        mid = self.counter
        
        obj['counter_id'] = mid
        self.memory_list.append(obj)
        
        # Document index: Extract text features (even if the observation includes an image)
        text = obj.get('text', '') if isinstance(obj, dict) else str(obj) if isinstance(obj, str) else ''
        if text or not isinstance(obj, dict) or 'image' not in obj:
            doc_embedding = self.encoder.encode_text(text, return_type='tensor')
            self.document_index[mid] = doc_embedding
        elif isinstance(obj, dict) and 'image' in obj:
            doc_embedding = self.encoder.encode_text(' ', return_type='tensor')
            self.document_index[mid] = doc_embedding
        
        # Image index: If an image is included, extract image features.
        if isinstance(obj, dict) and 'image' in obj and obj['image']:
            image_path = obj['image'].get('path') if isinstance(obj['image'], dict) else obj['image']
            if image_path:
                img_embedding = self.encoder.encode_image(image_path, return_type='tensor')
                self.image_index[mid] = img_embedding
        
        self.counter += 1
    
    def get_memory_element_by_mid(self, mid):
        return self.memory_list[mid]
    
    def get_memory_text_by_mid(self, mid):
        obj = self.memory_list[mid]
        if 'text' in obj:
            text = obj['text']
            return text if isinstance(text, str) else ' '.join(text)
        return ''
    
    def get_memory_image_by_mid(self, mid):
        """Get image information of a specified memory"""
        obj = self.memory_list[mid]
        return obj.get('image', None)
    
    def retrieve_by_query(self, query, modality, top_k=5):
        """
        Retrieve based on query and modality (simplified version)

        Args:
            query: Query object (dict or str)
            modality: 'no', 'document', or 'image'
            top_k: Return top-k results

        Returns:
            (retrieved_mids, scores): List of retrieved mids and scores
        """
        if modality == 'no':
            return [], []
        
        # Encode query based on modality
        if modality == 'image':
            # Image retrieval: only extract image features from query
            if isinstance(query, dict) and 'image' in query and query['image']:
                image_path = query['image'].get('path') if isinstance(query['image'], dict) else query['image']
                if image_path:
                    query_embedding = self.encoder.encode_image(image_path, return_type='tensor')
                else:
                    # If no image path, use text query (fallback)
                    text = query.get('text', '') if isinstance(query, dict) else str(query)
                    query_embedding = self.encoder.encode_text(text, return_type='tensor')
            else:
                # No image in query, use text query (fallback)
                text = query.get('text', '') if isinstance(query, dict) else str(query)
                query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.image_index
        else:  # document
            # Document retrieval: only extract text features from query
            text = query.get('text', '') if isinstance(query, dict) else str(query)
            query_embedding = self.encoder.encode_text(text, return_type='tensor')
            features = self.document_index
        
        if not features:
            return [], []
        
        # Normalize query vector
        query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarities = {}
        for mid, feat in features.items():
            feat_norm = feat / feat.norm(dim=-1, keepdim=True)
            sim = torch.matmul(query_norm, feat_norm.T).item()
            similarities[mid] = sim
        
        # If modality is 'image' and image_index results are insufficient, supplement from document_index
        if modality == 'image' and len(similarities) < top_k and self.document_index:
            # Use text query to supplement from document_index
            text = query.get('text', '') if isinstance(query, dict) else str(query)
            text_query_embedding = self.encoder.encode_text(text, return_type='tensor')
            text_query_norm = text_query_embedding / text_query_embedding.norm(dim=-1, keepdim=True)

            for mid, feat in self.document_index.items():
                feat_norm = feat / feat.norm(dim=-1, keepdim=True)
                text_sim = torch.matmul(text_query_norm, feat_norm.T).item()

                if mid not in similarities:
                    # If not in image_index, directly add the text feature score
                    similarities[mid] = text_sim
                else:
                    # If already in image_index, merge scores (average, since both image and text are relevant)
                    # This can boost the ranking of observations that match both image and text
                    image_sim = similarities[mid]
                    similarities[mid] = (image_sim + text_sim) / 2.0
        
        # Sort and return top-k
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        retrieved_mids = [mid for mid, _ in sorted_items]
        scores = [score for _, score in sorted_items]
        
        return retrieved_mids, scores

