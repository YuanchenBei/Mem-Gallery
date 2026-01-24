from abc import ABC, abstractmethod
from memengine.function import *
import torch
import numpy as np
from memengine.function.FactExtractor import SimpleFactExtractor, LLMFactExtractor

def __store_convert_str_to_observation__(method):
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper

class BaseStore(ABC):
    def __init__(self, config):
        self.config = config

    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass
    
    @ abstractmethod
    def __call__(self, observation):
        pass

class FUMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        self.storage.add(observation)

class MMFUMemoryStore(BaseStore):
    """
    MultiModal Full Memory Store:
    Stores multimodal observations (text + images) without any filtering.
    Similar to FUMemoryStore but designed for multimodal data.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        # Store the observation as-is (preserves text and image data)
        self.storage.add(observation)

class STMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.time_retrieval = kwargs['time_retrieval']

    def reset(self):
        pass
    
    @__store_convert_str_to_observation__
    def __call__(self, observation):
        if 'time' not in observation:
            timestamp = self.storage.counter
        else:
            timestamp = observation['time']
        self.storage.add(observation)
        self.time_retrieval.add(timestamp)

class LTMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        text = observation['text']
        self.storage.add(observation)
        self.text_retrieval.add(text)

class GAMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.text_retrieval = kwargs['text_retrieval']
        self.time_retrieval = kwargs['time_retrieval']
        self.importance_retrieval = kwargs['importance_retrieval']
        self.imporatance_judge = kwargs['imporatance_judge']
        self.reflector = kwargs['reflector']

    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        '''
        if 'time' not in observation:
            timestamp = self.storage.counter
        else:
            timestamp = observation['time']
        '''
        # Prefer using timestamp field (consistent with data input format)
        if 'timestamp' in observation:
            # Keep original timestamp for display
            # Also set time field to counter (for time_retrieval, requires numeric value)
            if 'time' not in observation:
                observation['time'] = self.storage.counter
            # time_retrieval uses counter (numeric), not timestamp (may be string)
            time_value = observation['time']
        elif 'time' not in observation:
            # If neither timestamp nor time exists, use counter as timestamp
            timestamp = self.storage.counter
            observation['timestamp'] = timestamp
            observation['time'] = timestamp
            time_value = timestamp
        else:
            # If only time field exists, copy to timestamp (for consistency)
            timestamp = observation['time']
            observation['timestamp'] = timestamp
            time_value = timestamp
        
        text = observation['text']

        # Judge importance score for the observation.
        importance_score = self.imporatance_judge({'message': text})

        if 'source' not in observation:
            observation['source'] = False

        # Take a reflection update immediately after storing the observation.
        self.reflector.add_reflection(importance_score, observation['source'])

        self.storage.add(observation)
        #self.time_retrieval.add(timestamp)
        self.time_retrieval.add(time_value)  # Use numeric value (counter or time)
        self.text_retrieval.add(text)
        self.importance_retrieval.add(importance_score)
          

class MGMemoryStore(BaseStore):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.main_context = kwargs['main_context']
        self.recall_storage = kwargs['recall_storage']
        self.recall_retrieval = kwargs['recall_retrieval']
        self.truncation = kwargs['truncation']

        self.summarizer = eval(self.config.summarizer.method)(self.config.summarizer)
        self.flush_checker = eval(self.config.flush_checker.method)(self.config.flush_checker)
    
    def reset(self):
        self.__reset_objects__([self.summarizer, self.flush_checker])

    def __flush_queue__(self):
        """
        Flush the FIFO queue.
        """
        FIFO_queue = self.main_context['FIFO_queue'].get_all_memory_in_order()

        flush_context = ''
        for mid, element in enumerate(FIFO_queue):
            flush_context += '\n%s' % element['text']
            self.recall_storage.add(element)
            self.recall_retrieval.add(element['text'])
            if self.flush_checker.check_truncation_needed(flush_context):
                break
        
        self.main_context['recursive_summary']['global'] = self.summarizer({
            'recursive_summary': self.main_context['recursive_summary']['global'],
            'flush_context': flush_context
        })

        self.main_context['FIFO_queue'].clear_memory(start=0, end=mid+1)

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        '''
        if 'time' not in observation:
            timestamp = self.main_context['FIFO_queue'].counter
            observation['time'] = timestamp
        else:
            timestamp = observation['time']
        '''
        # Prefer using timestamp field (consistent with data input format)
        if 'timestamp' in observation:
            # Keep original timestamp for display
            # Also set time field to counter (for possible numeric operations, though MGMemory doesn't use time_retrieval)
            if 'time' not in observation:
                observation['time'] = self.main_context['FIFO_queue'].counter
        elif 'time' not in observation:
            # If neither timestamp nor time exists, use counter as timestamp
            timestamp = self.main_context['FIFO_queue'].counter
            observation['timestamp'] = timestamp
            observation['time'] = timestamp
        else:
            # If only time field exists, copy to timestamp (for consistency)
            timestamp = observation['time']
            observation['timestamp'] = timestamp
        
        text = observation['text']
        
        self.main_context['FIFO_queue'].add(observation)

        # Check the state of FIFO queue to determine whether needs to flush it.
        if self.flush_checker.check_truncation_needed(text):
            self.__flush_queue__()


class MMMemoryStore(BaseStore):
    """MultiModal Memory Store"""
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
    
    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        """
        Store multimodal observation
        observation can be:
        - str: "text only"
        - dict: {'text': '...'}
        - dict: {'image': '/path/to/image.jpg'}
        - dict: {'text': '...', 'image': '/path/to/image.jpg'}
        """
        # Store raw data to Storage
        self.storage.add(observation)

        # Encode and store to MultiModalRetrieval
        self.multimodal_retrieval.add(observation)


class NGMemoryStore(BaseStore):
    """
    Neural Graph Memory Store:
    Stores multimodal events as graph nodes and creates edges based on relationships
    (temporal succession, semantic similarity, co-location)
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']  # GraphStorage
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.max_edges_per_node = getattr(config, 'max_edges_per_node', 5)

        # Time decay parameters (matching official implementation)
        self.temporal_decay_constant = getattr(config, 'temporal_decay_constant', 3600)  # 1 hour in seconds

    def reset(self):
        pass

    def __calculate_similarity__(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        import torch
        embedding1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
        embedding2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
        return torch.matmul(embedding1_norm, embedding2_norm.T).item()
    
    def __calculate_temporal_weight__(self, source_node_id, target_node_id):
        """
        Calculate temporal weight: uses exponential decay, closer in time means higher weight
        Matches official implementation: np.exp(-time_diff / 3600)
        """
        import numpy as np
        from datetime import datetime
        
        try:
            source_data = self.storage.get_memory_element_by_node_id(source_node_id)
            target_data = self.storage.get_memory_element_by_node_id(target_node_id)
            
            source_time = source_data.get('timestamp')
            target_time = target_data.get('timestamp')
            
            if source_time and target_time:
                # Handle different timestamp formats
                if isinstance(source_time, str) and isinstance(target_time, str):
                    # ISO format timestamp
                    try:
                        source_dt = datetime.fromisoformat(source_time.replace('Z', '+00:00'))
                        target_dt = datetime.fromisoformat(target_time.replace('Z', '+00:00'))
                        time_diff = abs((target_dt - source_dt).total_seconds())
                    except:
                        # Fallback: use node counter difference
                        time_diff = abs(target_data.get('mid', 0) - source_data.get('mid', 0))
                elif isinstance(source_time, (int, float)) and isinstance(target_time, (int, float)):
                    # Numeric timestamps
                    time_diff = abs(target_time - source_time)
                else:
                    # Use node counter as proxy
                    time_diff = abs(target_data.get('mid', 0) - source_data.get('mid', 0))
                
                # Exponential decay: closer in time = higher weight
                # Official implementation uses: np.exp(-time_diff / 3600)  # 1-hour decay constant
                return np.exp(-time_diff / self.temporal_decay_constant)
        except Exception as e:
            # If calculation fails, return default weight
            pass
        
        return 1.0
    
    def __create_edges__(self, new_node_id, new_embedding):
        """
        Create edges for new node:
        1. Create temporal edge with most recently added node (temporal succession)
        2. Create similarity edges with semantically similar nodes (semantic similarity)
        """
        if self.storage.get_element_number() <= 1:
            return
        
        # Get all existing nodes (in insertion order)
        all_node_ids = [node_id for node_id in self.storage.memory_order_map if node_id != new_node_id]

        if not all_node_ids:
            return

        # 1. Create temporal edge: connect to most recently added node (last node)
        # Use time decay weight (matching official implementation)
        if len(all_node_ids) > 0:
            last_node_id = all_node_ids[-1]
            temporal_weight = self.__calculate_temporal_weight__(last_node_id, new_node_id)
            
            self.storage.add_edge(
                last_node_id, 
                new_node_id, 
                {
                    'type': 'temporal_succession',
                    'weight': float(temporal_weight)
                }
            )
        
        # 2. Create semantic similarity edges: connect to most similar nodes
        similarities = []
        for node_id in all_node_ids:
            # Get node embedding (need to retrieve from multimodal_retrieval)
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if (self.multimodal_retrieval.tensorstore is not None and
                node_mid < self.multimodal_retrieval.tensorstore.size(0)):
                # Calculate similarity
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                similarity = self.__calculate_similarity__(new_embedding, node_embedding)
                similarities.append((node_id, similarity))

        # Sort by similarity, select most similar nodes to create edges
        similarities.sort(key=lambda x: x[1], reverse=True)
        for node_id, similarity in similarities[:self.max_edges_per_node]:
            if similarity >= self.similarity_threshold:
                self.storage.add_edge(
                    node_id,
                    new_node_id,
                    {
                        'type': 'semantic_similarity',
                        'weight': float(similarity)
                    }
                )

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        """
        Store multimodal observation as a graph node
        """
        # Add timestamp (if not present)
        if 'timestamp' not in observation:
            observation['timestamp'] = self.storage.node_counter

        # Add node to graph storage
        node_id = self.storage.add_node(observation)

        # Encode observation and add to multimodal_retrieval
        embedding = self.multimodal_retrieval.add(observation)

        # Create edges (connect to other nodes)
        self.__create_edges__(node_id, embedding)


class AUGUSTUSMemoryStore(BaseStore):
    """
    AUGUSTUS Memory Store:
    Stores observations in Contextual Memory (concept-tagged graph).
    Creates edges based on temporal succession, semantic similarity, and concept associations.
    
    As per AUGUSTUS paper:
    - Contextual Memory: Stores concept-tagged graph nodes with multimodal context
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs.get('contextual_memory') or kwargs.get('storage')  # TagGraphStorage for contextual memory
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        self.concept_extractor = kwargs['concept_extractor']
        
        # If only storage is provided (backward compatibility), use it as contextual_memory
        if 'contextual_memory' not in kwargs and 'storage' in kwargs:
            self.storage = kwargs['storage']
        
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.max_edges_per_node = getattr(config, 'max_edges_per_node', 5)
        self.min_shared_concepts = getattr(config, 'min_shared_concepts', 1)
        
        # Time decay parameters
        self.temporal_decay_factor = getattr(config, 'temporal_decay_factor', 0.1)
        self.enable_time_decay = getattr(config, 'enable_time_decay', True)
    
    def reset(self):
        if hasattr(self.concept_extractor, 'reset'):
            self.concept_extractor.reset()
    
    def __calculate_similarity__(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        import torch
        embedding1_norm = torch.nn.functional.normalize(embedding1, dim=-1)
        embedding2_norm = torch.nn.functional.normalize(embedding2, dim=-1)
        return torch.matmul(embedding1_norm, embedding2_norm.T).item()

    def __calculate_time_decay__(self, timestamp1, timestamp2):
        """
        Calculate time decay factor between two timestamps.
        Returns a value between 0 and 1, where 1 means no decay (same time).
        """
        if not self.enable_time_decay:
            return 1.0
        
        try:
            if isinstance(timestamp1, (int, float)) and isinstance(timestamp2, (int, float)):
                time_diff = abs(timestamp1 - timestamp2)
            else:
                # If timestamps are not numeric, use node counter as proxy
                time_diff = abs(self.storage.node_counter - max(timestamp1, timestamp2))
            
            # Exponential decay: decay_factor controls decay rate
            import numpy as np
            decay = np.exp(-self.temporal_decay_factor * time_diff / 100.0)
            return max(0.1, min(1.0, decay))  # Clamp between 0.1 and 1.0
        except:
            return 1.0
    
    def __extract_concepts__(self, observation):
        """Extract concepts from observation."""
        return self.concept_extractor.extract(observation)
    
    def __create_edges__(self, new_node_id, new_embedding, new_concepts):
        """
        Create edges for new node:
        1. Create temporal edge with most recently added node (temporal succession)
        2. Create similarity edges with semantically similar nodes (semantic similarity)
        3. Create concept association edges with nodes sharing concepts (concept association)
        """
        if self.storage.get_element_number() <= 1:
            return

        # Get all existing nodes (in insertion order)
        all_node_ids = [node_id for node_id in self.storage.memory_order_map if node_id != new_node_id]

        if not all_node_ids:
            return

        # 1. Create temporal edge: connect to most recently added node (last node)
        if len(all_node_ids) > 0:
            last_node_id = all_node_ids[-1]
            # Get timestamps for time decay calculation
            new_timestamp = self.storage.get_memory_element_by_node_id(new_node_id).get('timestamp', self.storage.node_counter)
            last_timestamp = self.storage.get_memory_element_by_node_id(last_node_id).get('timestamp', self.storage.node_counter)
            time_decay = self.__calculate_time_decay__(new_timestamp, last_timestamp)
            
            self.storage.add_edge(
                last_node_id, 
                new_node_id, 
                {
                    'type': 'temporal_succession',
                    'weight': 1.0 * time_decay  # Apply time decay
                }
            )
        
        # 2. Create semantic similarity edges: connect to most similar nodes
        similarities = []
        for node_id in all_node_ids:
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if (self.multimodal_retrieval.tensorstore is not None and
                node_mid < self.multimodal_retrieval.tensorstore.size(0)):
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                similarity = self.__calculate_similarity__(new_embedding, node_embedding)
                similarities.append((node_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        for node_id, similarity in similarities[:self.max_edges_per_node]:
            if similarity >= self.similarity_threshold:
                # Apply time decay to semantic similarity edges
                node_timestamp = self.storage.get_memory_element_by_node_id(node_id).get('timestamp', self.storage.node_counter)
                new_timestamp = self.storage.get_memory_element_by_node_id(new_node_id).get('timestamp', self.storage.node_counter)
                time_decay = self.__calculate_time_decay__(new_timestamp, node_timestamp)

                # Weighted similarity with time decay
                final_weight = float(similarity) * time_decay

                self.storage.add_edge(
                    node_id,
                    new_node_id,
                    {
                        'type': 'semantic_similarity',
                        'weight': final_weight
                    }
                )

        # 3. Create concept association edges: connect to nodes sharing concepts
        new_concepts_set = set(new_concepts)
        # Calculate concept importance (TF-like: more shared concepts = higher importance)
        total_concepts_in_graph = len(self.storage.concept_index) if hasattr(self.storage, 'concept_index') else 1
        
        for node_id in all_node_ids:
            node_concepts = self.storage.get_concepts_by_node(node_id)
            shared_concepts = new_concepts_set & node_concepts
            
            if len(shared_concepts) >= self.min_shared_concepts:
                # Calculate concept importance weight
                # Weight = (shared_concepts / max(query_concepts, node_concepts)) * concept_rarity_factor
                max_concepts = max(len(new_concepts_set), len(node_concepts), 1)
                concept_overlap_ratio = len(shared_concepts) / max_concepts
                
                # Concept rarity factor: less common concepts are more important
                # (simplified: use inverse of concept frequency in graph)
                concept_rarity = 1.0
                if total_concepts_in_graph > 0:
                    # Average rarity: concepts that appear in fewer nodes are more important
                    avg_concept_frequency = sum(len(nodes) for nodes in self.storage.concept_index.values()) / total_concepts_in_graph
                    concept_rarity = 1.0 / (1.0 + avg_concept_frequency / 10.0)  # Normalize
                
                # Apply time decay
                node_timestamp = self.storage.get_memory_element_by_node_id(node_id).get('timestamp', self.storage.node_counter)
                new_timestamp = self.storage.get_memory_element_by_node_id(new_node_id).get('timestamp', self.storage.node_counter)
                time_decay = self.__calculate_time_decay__(new_timestamp, node_timestamp)
                
                # Final weight: combines overlap ratio, rarity, and time decay
                concept_weight = concept_overlap_ratio * (1.0 + concept_rarity) * time_decay
                
                self.storage.add_edge(
                    node_id,
                    new_node_id,
                    {
                        'type': 'concept_association',
                        'shared_concepts': list(shared_concepts),
                        'weight': concept_weight
                    }
                )

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        """
        Store observation in Contextual Memory (concept-tagged graph).
        """
        # Prepare observation copy for contextual memory
        contextual_obs = observation.copy() if isinstance(observation, dict) else {'text': str(observation)}
        
        # Extract concepts from observation
        concepts = self.__extract_concepts__(observation)
        contextual_obs['concepts'] = concepts
        
        # Add timestamp if not present
        if 'timestamp' not in contextual_obs:
            contextual_obs['timestamp'] = self.storage.node_counter
        
        # Add node to graph storage (TagGraphStorage will handle concepts)
        node_id = self.storage.add_node(contextual_obs)
        
        # Encode observation and add to multimodal retrieval
        embedding = self.multimodal_retrieval.add(contextual_obs)
        
        # Create edges (temporal, semantic, and concept-based)
        self.__create_edges__(node_id, embedding, concepts)


class UniversalRAGStore(BaseStore):
    """
    UniversalRAG Store
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.storage = kwargs['storage']  # UniversalRAGStorage

    def reset(self):
        pass

    @__store_convert_str_to_observation__
    def __call__(self, observation):
        # Store and automatically extract features
        self.storage.add(observation)