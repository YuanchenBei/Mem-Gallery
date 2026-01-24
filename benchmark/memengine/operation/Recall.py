from abc import ABC, abstractmethod
from memengine.function import *

def __recall_convert_str_to_observation__(method):
    """
    If the input is a string, convert it to the dict form.
    """
    def wrapper(self, observation):
        if isinstance(observation, str):
            return method(self, {'text': observation})
        else:
            return method(self, observation)
    return wrapper

class BaseRecall(ABC):
    def __init__(self, config):
        self.config = config
    
    def __reset_objects__(self, objects):
        for obj in objects:
            obj.reset()
    
    @abstractmethod
    def reset(self):
        pass

    @ abstractmethod
    def __call__(self, query):
        pass

class FUMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization(self.storage.get_all_memory_in_order())
        return self.truncation(memory_context)

class MMFUMemoryRecall(BaseRecall):
    """
    MultiModal Full Memory Recall:
    Retrieves all memories (text + images) and truncates based on token budget.
    Unlike FUMemoryRecall which only handles text, this preserves multimodal data.
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Get all memories in order (includes both text and image data)
        all_memories = self.storage.get_all_memory_in_order()
        
        # Apply multimodal truncation (based on token budget: text + image tokens)
        truncated_memories = self.truncation(all_memories)
        
        # Expose retrieved ids for evaluation
        self.last_retrieved_ids = []
        for mem in truncated_memories:
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                self.last_retrieved_ids.append(mem['dialogue_id'])
        
        # Use MultiModalUtilization to format the result
        result = self.utilization(truncated_memories)
        
        return result

class STMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.time_retrieval])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Get the most recent information.
        ranking_ids = self.time_retrieval(query['text'])
        # Expose retrieved ids for evaluation
        try:
            elements = [self.storage.get_memory_element_by_mid(mid) for mid in ranking_ids]
            self.last_retrieved_ids = []
            for mem in elements:
                if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                    self.last_retrieved_ids.append(mem['dialogue_id'])
        except Exception:
            self.last_retrieved_ids = []
        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization(elements)

        return self.truncation(memory_context)


class LTMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        # Retrieval process.
        ranking_ids = self.text_retrieval(query['text'])
        #memory_context = self.utilization([self.storage.get_memory_text_by_mid(mid) for mid in ranking_ids])
        elements = [self.storage.get_memory_element_by_mid(mid) for mid in ranking_ids]
        # Expose evaluation IDs
        self.last_retrieved_ids = []
        for mem in elements:
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                self.last_retrieved_ids.append(mem['dialogue_id'])
        memory_context = self.utilization(elements)

        return self.truncation(memory_context)


class GAMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.text_retrieval = eval(self.config.text_retrieval.method)(self.config.text_retrieval)
        self.time_retrieval = eval(self.config.time_retrieval.method)(self.config.time_retrieval)
        self.importance_retrieval = eval(self.config.importance_retrieval.method)(self.config.importance_retrieval)
        
        self.imporatance_judge = eval(self.config.importance_judge.method)(self.config.importance_judge)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.text_retrieval, self.time_retrieval, self.importance_retrieval, self.imporatance_judge])

    def __retention__(self, target_ids, timestamp):
        """Update the memory of `target_ids` with their recency to `timestamp`.

        Args:
            target_ids (list): the target ids whose memory recency should be updated.
            timestamp (int/float): current timestamp.
        """
        for index in target_ids:
            self.time_retrieval.update(index, timestamp)

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            return self.config.empty_memory
        
        text = query['text']
        # Prefer timestamp field, but time_retrieval needs numeric values
        # If timestamp is a string, use time field or counter
        if 'timestamp' in query:
            # If timestamp is a string, use time field (numeric) or counter
            if 'time' in query:
                timestamp = query['time']  # Prefer time (numeric value)
            else:
                timestamp = self.storage.counter  # Use counter as numeric value
        elif 'time' not in query:
            timestamp = self.storage.counter
        else:
            timestamp = query['time']
        
        # Calculate weighted retrieval scores.
        text_scores, _ = self.text_retrieval(text, topk=False, with_score = True, sort = False)
        recency_scores, _ = self.time_retrieval(timestamp, topk=False, with_score = True, sort = False)
        importance_scores, _ = self.importance_retrieval(None, topk=False, with_score = True, sort = False)

        score_metrix = torch.stack([text_scores, recency_scores, importance_scores], dim=1)
        comprehensive_scores = torch.matmul(score_metrix, torch.ones(3).to(self.text_retrieval.encoder.device))

        scores, ranking_ids = torch.sort(comprehensive_scores, descending=True)

        if hasattr(self.config, 'topk'):
            scores, ranking_ids = scores[:self.config.topk], ranking_ids[:self.config.topk]
        
        # Update the recency of retrieved memory.
        self.__retention__(ranking_ids, timestamp)

        # Expose retrieved ids for evaluation
        try:
            elements = [self.storage.get_memory_element_by_mid(int(mid)) for mid in ranking_ids]
            self.last_retrieved_ids = []
            for mem in elements:
                if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                    self.last_retrieved_ids.append(mem['dialogue_id'])
        except Exception:
            self.last_retrieved_ids = []

        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization(elements)

        return self.truncation(memory_context)


class MGMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

        self.main_context = kwargs['main_context']
        self.recall_storage = kwargs['recall_storage']
        self.archival_storage = kwargs['archival_storage']

        self.recall_retrieval = eval(self.config.recall_retrieval.method)(self.config.recall_retrieval)
        self.archival_retrieval = eval(self.config.archival_retrieval.method)(self.config.archival_retrieval)
        self.trigger = LLMTrigger(self.config.trigger)
        self.warning_number = int(self.config.warning_threshold * self.config.truncation.number)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.recall_retrieval, self.archival_retrieval, self.trigger])

    def __get_current_memory_prompt(self):
        """
        To provide the current state of memory model for constructing prompts.
        """
        piece_num, piece_mode = self.truncation.get_piece_number(self.get_current_memory_context()), self.config.truncation.mode
        working_memory = '\n'.join(['[%d] %s' % (wid, wtext['text']) for wid, wtext in enumerate(self.main_context['working_context'].get_all_memory_in_order())])
        FIFO_memory = '\n'.join(['[%d] %s' % (fid, ftext['text']) for fid, ftext in enumerate(self.main_context['FIFO_queue'].get_all_memory_in_order())])
        prompt = """Total Capacity: %s %ss
Working Memory(capacity %s words):
%s
Recursive Memory Summary:
%s
FIFO Memory:
%s
""" % (piece_num, piece_mode, self.config.truncation.number, working_memory, self.main_context['recursive_summary']['global'], FIFO_memory)

        return prompt

    def get_current_memory_context(self):
        # Pass complete memory elements (including timestamp) instead of just text
        memory_context = self.utilization({
            'Working Memory': self.main_context['working_context'].get_all_memory_in_order(),
            'Recursive Memory Summary': self.main_context['recursive_summary']['global'],
            'FIFO Memory': self.main_context['FIFO_queue'].get_all_memory_in_order()
        })
        return memory_context

    def __trigger_function_call__(self, text):
        """Trigger extensible functions.

        Args:
            text (str): query or observation.
        """
        warning_flag = self.truncation.get_piece_number(text) > self.warning_number

        info_dict = {}
        
        info_dict['text'] = text
        info_dict['memory_prompt'] = self.__get_current_memory_prompt()

        # Insert warning sentences.
        if warning_flag:
            info_dict['warning_content'] = self.config.warning_content + '\n'
        else:
            info_dict['warning_content'] = ''

        # Allow trigger none function.
        if hasattr(self.config.trigger, 'no_execute'):
            info_dict['no_execute_prompt'] = self.config.trigger.no_execute + '\n'
        else:
            info_dict['no_execute_prompt'] = ''

        # Add few-shot examples.
        info_dict['few_shot'] = self.config.trigger.few_shot

        # Get trigger results, parse and execute them.
        exe_list = self.trigger(info_dict)
        for (func_name, func_args) in exe_list:
            try:
                execute_command = 'self.__%s__' % func_name
                eval(execute_command)(*func_args)
                print('Successfully execute Function [%s(%s)]' % (func_name, func_args))
            except Exception as e:
                print('Fail to execute Function [%s(%s)]: %s'% (func_name, func_args, e))

    def __memory_retrieval__(self, query):
        """Retrieve query-related information from (external) archival storage, and add the result into working memory.

        Args:
            query (str): a string to retrieve relevant information (e.g., \'Alice\'s name).

        Returns:
            list: list of indexes to be added into working memory.
        """
        if self.archival_storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.archival_retrieval(text)
        for mid in ranking_ids:
            self.main_context['working_context'].add(self.archival_storage.get_memory_element_by_mid(mid))
    
    def __memory_recall__(self, query):
        """retrieve query-related information from (external) recall storage, and add the result into FIFO memory.

        Args:
            query (str): a string to retrieve relevant information (e.g., \'Alice\'s name).

        Returns:
            list: list of indexes to be added into FIFO memory.
        """
        if self.recall_storage.is_empty():
            return self.config.empty_memory
        text = query['text']
        ranking_ids = self.recall_retrieval(text)
        for mid in ranking_ids:
            self.main_context['FIFO_queue'].add(self.recall_storage.get_memory_element_by_mid(mid))

    def __memory_archive__(self, memory_list):
        """Archive some memory from FIFO memory into (external) archival storage.

        Args:
            memory_list (list): the index list of FIFO memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.recall_storage.add(element)
            self.recall_retrieval.add(element['text'])

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_transfer__(self, memory_list):
        """Transfer some memory from FIFO memory into working memory.

        Args:
            memory_list (list): the index list of FIFO memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['FIFO_queue'].get_memory_element_by_mid(mid)
            self.main_context['working_context'].add(element)

        self.main_context['FIFO_queue'].delete_by_mid_list(memory_list)
    
    def __memory_save__(self, memory_list):
        """Archive some memory from working memory into (external) archival storage.

        Args:
            memory_list (list): the index list of working memory (e.g., [0, 2, 3]).
        """
        for mid in memory_list:
            element = self.main_context['working_context'].get_memory_element_by_mid(mid)
            self.archival_storage.add(element)
            self.archival_retrieval.add(element['text'])

        self.main_context['working_context'].delete_by_mid_list(memory_list)
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        text = query['text']

        # Trigger functions to update the current state of memory model.
        self.__trigger_function_call__(text)

        return self.get_current_memory_context()
    
class RFMemoryRecall(BaseRecall):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.insight = kwargs['insight']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)

    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization])

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            if self.insight['global_insight']:
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight']
                })
            else:
                return self.config.empty_memory
        else:
            if self.insight['global_insight']:
                # Pass complete memory elements (including timestamp) instead of just text
                memory_context = self.utilization({
                    'Insight': self.insight['global_insight'],
                    'Memory': self.storage.get_all_memory_in_order()
                })
            else:
                # Pass complete memory elements (including timestamp) instead of just text
                memory_context = self.utilization(self.storage.get_all_memory_in_order())
        
        return self.truncation(memory_context)


class MMMemoryRecall(BaseRecall):
    """MultiModal Memory Recall"""
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        """
        Recall memories based on multimodal query
        Supports two types of Utilization:
        1. ConcateUtilization: Returns text string
        2. MultiModalUtilization: Returns {'text': str, 'images': list}
        """
        if self.storage.is_empty():
            # Record empty retrieval ID list
            self.last_retrieved_ids = []
            return []
        
        # Multimodal retrieval: return the most relevant memory indices
        ranking_ids = self.multimodal_retrieval(query)
        
        if len(ranking_ids) == 0:
            #empty_result = self.config.empty_memory
            self.last_retrieved_ids = []
            return []
        
        # Collect memories (containing text and image information)
        memories = []
        retrieved_ids = []
        for mid in ranking_ids:
            mem = self.storage.get_memory_element_by_mid(int(mid))
            memories.append(mem)
            # Collect alignment IDs for evaluation
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                retrieved_ids.append(mem['dialogue_id'])
        # Expose for evaluation
        self.last_retrieved_ids = retrieved_ids

        # Format using Utilization
        result = self.utilization(memories)


        # MultiModalUtilization returns dict
        # Apply truncation to text part
        #result['text'] = self.truncation(result['text'])
        return result


class NGMemoryRecall(BaseRecall):
    """
    Neural Graph Memory Recall:
    Query-aware graph traversal for memory retrieval
    Combines embedding similarity with graph structure
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs['storage']  # GraphStorage
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        
        # Graph traversal parameters
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        
        # Traversal strategy: 'depth_first' (recursive with depth limit) or 'breadth_first' (expand all neighbors)
        # 'breadth_first' matches the original Neural-Graph-Memory-NGM implementation
        self.traversal_strategy = getattr(config, 'traversal_strategy', 'depth_first')
        
        # For breadth_first: initial candidate multiplier (original uses top_k * 2)
        self.initial_candidate_multiplier = getattr(config, 'initial_candidate_multiplier', 2)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval])
    
    def __graph_traversal_depth_first__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        Depth-first recursive graph traversal (original implementation)
        Performs query-aware graph traversal from start nodes with depth limit

        Args:
            query_embedding: Query embedding vector
            start_node_ids: List of start node IDs
            visited: Set of visited nodes
            depth: Current traversal depth

        Returns:
            list: List of traversed node IDs (sorted by relevance)
        """
        import torch
        
        if visited is None:
            visited = set()
        
        if depth >= self.max_depth:
            return []
        
        candidates = []
        
        for node_id in start_node_ids:
            if node_id in visited:
                continue
            
            visited.add(node_id)

            # Get node embedding
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

            # Calculate similarity with query
            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            similarity = torch.matmul(query_norm, node_norm.T).item()

            # If similarity exceeds threshold, add to candidates
            if similarity >= self.traversal_threshold:
                candidates.append((node_id, similarity))

            # Continue traversing neighbor nodes
            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal_depth_first__(
                    query_embedding, 
                    neighbors, 
                    visited, 
                    depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    def __graph_traversal_breadth_first__(self, query_embedding, initial_node_ids):
        """
        Breadth-first expansion graph traversal (matching original Neural-Graph-Memory-NGM implementation)
        Expands all neighbors of initial candidate nodes, then recalculates similarity

        Args:
            query_embedding: Query embedding vector
            initial_node_ids: List of initial candidate node IDs

        Returns:
            list: List of (node_id, similarity) tuples
        """
        import torch

        # 1. Expand search: add all initial candidate nodes and their neighbors
        expanded_nodes = set()
        for node_id in initial_node_ids:
            expanded_nodes.add(node_id)
            # Add all neighbor nodes
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor in neighbors:
                expanded_nodes.add(neighbor)

        # 2. Recalculate similarity for all expanded nodes
        final_similarities = []
        for node_id in expanded_nodes:
            try:
                node_mid = self.storage.get_mid_by_node_id(node_id)
                if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                    continue
                
                node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]

                # Calculate cosine similarity
                query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                similarity = torch.matmul(query_norm, node_norm.T).item()
                
                # Apply degree boosting (matching official implementation)
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                final_similarities.append((node_id, boosted_similarity))
            except (KeyError, IndexError, Exception):
                # If node does not exist or error occurs, skip
                continue

        # 3. Sort by similarity
        final_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return final_similarities
    
    def __graph_traversal__(self, query_embedding, start_node_ids, visited=None, depth=0):
        """
        Graph traversal entry method, selects depth-first or breadth-first strategy based on config

        Args:
            query_embedding: Query embedding vector
            start_node_ids: List of start node IDs
            visited: Set of visited nodes (only for depth-first)
            depth: Current traversal depth (only for depth-first)

        Returns:
            list: List of traversed node IDs (sorted by relevance)
        """
        if self.traversal_strategy == 'breadth_first':
            # Breadth-first: expand all neighbors
            return self.__graph_traversal_breadth_first__(query_embedding, start_node_ids)
        else:
            # Depth-first: recursive traversal (default)
            return self.__graph_traversal_depth_first__(query_embedding, start_node_ids, visited, depth)
    
    def __deep_reasoning_expand__(self, query_embedding, query_concepts_set, query_entities,
                                  query_timestamp, seed_nodes):
        """Optional multi-hop reasoning to surface deeper nodes."""
        from collections import deque

        if not seed_nodes:
            self.last_reasoning_paths = {}
            self.last_reasoning_details = {}
            return {}

        results = {}
        queue = deque()
        for node_id in seed_nodes:
            queue.append((node_id, 0, [node_id], 1.0))

        while queue:
            current_node, depth, path, path_score = queue.popleft()
            if depth >= self.deep_reasoning_max_hops:
                continue

            neighbors = self.storage.get_neighbors(current_node)
            if not neighbors:
                continue

            neighbor_evaluations = []
            for neighbor in neighbors:
                if neighbor in path:
                    continue
                evaluation = self.__evaluate_node_alignment__(
                    query_embedding,
                    query_concepts_set,
                    query_entities,
                    query_timestamp,
                    neighbor
                )
                if evaluation is None:
                    continue
                combined_score, score_details, _ = evaluation
                neighbor_evaluations.append((neighbor, combined_score, score_details))

            if not neighbor_evaluations:
                continue

            neighbor_evaluations.sort(key=lambda item: item[1], reverse=True)
            trimmed = neighbor_evaluations[:self.deep_reasoning_branch_factor]

            for neighbor, combined_score, score_details in trimmed:
                edge_info = self.storage.get_edges_from(current_node).get(neighbor, {})
                edge_weight = float(edge_info.get('weight', 0.5))
                relation_type = edge_info.get('type', 'generic')

                adjusted_score = (combined_score + edge_weight) / 2.0
                cumulative_score = path_score * adjusted_score
                new_path = path + [neighbor]

                if cumulative_score >= self.deep_reasoning_threshold:
                    current_record = results.get(neighbor)
                    if current_record is None or cumulative_score > current_record['score']:
                        results[neighbor] = {
                            'score': cumulative_score,
                            'details': score_details,
                            'path': new_path,
                            'relation': relation_type,
                            'edge_weight': edge_weight
                        }

                if depth + 1 < self.deep_reasoning_max_hops:
                    queue.append((neighbor, depth + 1, new_path, cumulative_score))

        self.last_reasoning_paths = {node_id: info['path'] for node_id, info in results.items()}
        self.last_reasoning_details = results
        return results

    @__recall_convert_str_to_observation__
    def __call__(self, query):
        """
        Recall memories using query-aware graph traversal
        """
        if self.storage.is_empty():
            return []

        # 1. First use embedding similarity to find the most relevant start nodes
        # For breadth-first, use more initial candidates (matching original implementation: top_k * multiplier)
        initial_topk = 3
        if self.traversal_strategy == 'breadth_first':
            initial_topk = min(initial_topk * self.initial_candidate_multiplier, self.storage.get_element_number())
        
        ranking_ids = self.multimodal_retrieval(query, topk=min(initial_topk, self.storage.get_element_number()))
        
        if len(ranking_ids) == 0:
            return []

        # 2. Encode query as embedding
        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)

        # 3. Start graph traversal from start nodes
        start_node_ids = []
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
            except (KeyError, IndexError):
                # If node ID does not exist, skip
                continue

        # Execute graph traversal (select depth-first or breadth-first based on strategy)
        traversal_results = self.__graph_traversal__(query_embedding, start_node_ids)

        # 4. Merge initial retrieval results and traversal results, deduplicate and sort
        # Note: breadth-first strategy has already applied degree boosting in traversal, depth-first needs to apply it here
        all_candidates = {}

        if self.traversal_strategy == 'breadth_first':
            # Breadth-first: traversal results already contain all nodes (initial + neighbors) and degree boosting
            # Directly use traversal results
            for node_id, boosted_similarity in traversal_results:
                all_candidates[node_id] = boosted_similarity
        else:
            # Depth-first: need to merge initial retrieval results and traversal results, and apply degree boosting
            import torch
            # Add initial retrieval results (give higher weight, apply degree boosting)
            for mid in ranking_ids:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                # Calculate similarity for initial nodes
                try:
                    node_mid = self.storage.get_mid_by_node_id(node_id)
                    if node_mid < self.multimodal_retrieval.tensorstore.size(0):
                        node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
                        query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
                        node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
                        similarity = torch.matmul(query_norm, node_norm.T).item()
                        
                        # Apply graph-based boosting (matching official implementation)
                        node_degree = self.storage.get_node_degree(node_id)
                        degree_boost = 1 + (node_degree * 0.1)
                        all_candidates[node_id] = similarity * degree_boost
                    else:
                        all_candidates[node_id] = 1.0
                except Exception:
                    all_candidates[node_id] = 1.0

            # Add traversal results (apply degree boosting)
            for node_id, similarity in traversal_results:
                # Apply graph-based boosting (matching official implementation)
                node_degree = self.storage.get_node_degree(node_id)
                degree_boost = 1 + (node_degree * 0.1)
                boosted_similarity = similarity * degree_boost
                
                if node_id not in all_candidates:
                    all_candidates[node_id] = boosted_similarity
                else:
                    # If already exists, take the larger value
                    all_candidates[node_id] = max(all_candidates[node_id], boosted_similarity)

        # 5. Sort by similarity, select top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            return []

        # 6. Collect memories (return all selected top-k nodes, not just the best node + neighbors)
        memories = []
        retrieved_ids = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                retrieved_ids.append(mem['dialogue_id'])

        # Expose for evaluation
        self.last_retrieved_ids = retrieved_ids

        # 7. Format using Utilization
        result = self.utilization(memories)
        return result


class AUGUSTUSMemoryRecall(BaseRecall):
    """
    AUGUSTUS Memory Recall:
    Implements hybrid retrieval with CoPe (Contextual-Personalized) search and embedding-based retrieval.
    Supports Contextual Memory (concept-tagged graph).
    
    Hybrid retrieval strategy:
    - CoPe Search Stage 1: Search over tags/concepts to find matching nodes
    - Embedding-based retrieval: Find relevant nodes by similarity (fallback/complement)
    - CoPe Search Stage 2: Graph traversal through detailed contexts attached to tags
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.storage = kwargs.get('contextual_memory') or kwargs.get('storage')  # TagGraphStorage for contextual memory
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
        self.multimodal_retrieval = kwargs['multimodal_retrieval']
        self.concept_extractor = kwargs['concept_extractor']
        self.concept_retrieval = kwargs['concept_retrieval']
        
        # If only storage is provided (backward compatibility), use it as contextual_memory
        if 'contextual_memory' not in kwargs and 'storage' in kwargs:
            self.storage = kwargs['storage']
        
        # Set storage for concept retrieval
        self.concept_retrieval.set_storage(self.storage)
        
        # Graph traversal parameters
        self.max_depth = getattr(config, 'max_depth', 3)
        self.max_nodes = getattr(config, 'max_nodes', 10)
        self.traversal_threshold = getattr(config, 'traversal_threshold', 0.5)
        
        # Fusion parameters
        self.embedding_weight = getattr(config, 'embedding_weight', 0.5)
        self.concept_weight = getattr(config, 'concept_weight', 0.5)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.multimodal_retrieval, self.concept_extractor, self.concept_retrieval])
    
    def cope_search(self, query):
        """
        CoPe Search Stage 1: Search over tags/concepts.
        
        This implements the first stage of CoPe search as per AUGUSTUS paper:
        - Extract concepts from query
        - Find nodes that match query concepts through tag-level search
        
        Stage 2 (graph traversal through detailed contexts) is implemented in __call__ method.
        
        Args:
            query: Query observation (dict with 'text' and/or 'image')
            
        Returns:
            list: List of node_ids matching the query through concept-based search
        """
        # Extract concepts from query
        query_concepts = self.concept_extractor.extract(query)
        
        if not query_concepts:
            return []
        
        # Search nodes that match query concepts (Stage 1: tag-level search)
        concept_counts = self.storage.get_nodes_by_concepts(query_concepts)
        
        # Filter nodes by minimum concept overlap
        matching_node_ids = []
        for node_id, overlap_count in concept_counts.items():
            if overlap_count >= self.concept_retrieval.min_concept_overlap:
                matching_node_ids.append(node_id)
        
        return matching_node_ids
    
    def __calculate_concept_score__(self, query_concepts_set, node_concepts):
        """
        Calculate concept matching score.
        """
        if not query_concepts_set or not node_concepts:
            return 0.0
        
        overlap = query_concepts_set & node_concepts
        return len(overlap) / max(len(query_concepts_set), len(node_concepts), 1)
    
    def __graph_traversal__(self, query_embedding, query_concepts, start_node_ids, visited=None, depth=0):
        """
        Graph traversal from start nodes (combining embedding and concept similarity).
        """
        import torch
        
        if visited is None:
            visited = set()
        
        if depth >= self.max_depth:
            return []
        
        candidates = []
        query_concepts_set = set(query_concepts) if query_concepts else set()
        
        for node_id in start_node_ids:
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Get node embedding for similarity calculation
            node_mid = self.storage.get_mid_by_node_id(node_id)
            if node_mid >= self.multimodal_retrieval.tensorstore.size(0):
                continue
            
            node_embedding = self.multimodal_retrieval.tensorstore[node_mid:node_mid+1]
            
            # Calculate embedding similarity
            query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
            node_norm = torch.nn.functional.normalize(node_embedding, dim=-1)
            embedding_similarity = torch.matmul(query_norm, node_norm.T).item()
            
            # Calculate concept score
            node_concepts = self.storage.get_concepts_by_node(node_id)
            concept_score = self.__calculate_concept_score__(query_concepts_set, node_concepts)
            
            # Combined score
            combined_score = (self.embedding_weight * embedding_similarity + 
                            self.concept_weight * concept_score)
            
            if combined_score >= self.traversal_threshold:
                candidates.append((node_id, combined_score))
            
            # Continue traversing neighbors
            neighbors = self.storage.get_neighbors(node_id)
            if neighbors:
                neighbor_candidates = self.__graph_traversal__(
                    query_embedding,
                    query_concepts,
                    neighbors, 
                    visited, 
                    depth + 1
                )
                candidates.extend(neighbor_candidates)
        
        return candidates
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        """
        Recall memories using hybrid retrieval with CoPe (Contextual-Personalized) search.
        """
        if self.storage.is_empty():
            self.last_retrieved_ids = []
            return []
        
        # 1. Embedding-based retrieval: find most relevant starting nodes
        ranking_ids = self.multimodal_retrieval(query, topk=min(3, self.storage.get_element_number()))
        
        # 2. Extract concepts from query (for CoPe Stage 1)
        query_concepts = self.concept_extractor.extract(query)
        
        # 3. CoPe Search Stage 1: Search over tags/concepts
        cope_node_ids = []
        if query_concepts:
            cope_node_ids = self.cope_search(query)
        
        # 4. Encode query as embedding
        query_embedding = self.multimodal_retrieval.encoder(query, return_type='tensor')
        if self.multimodal_retrieval.config.mode == 'cosine':
            query_embedding = self.multimodal_retrieval.__normalize__(query_embedding)
        
        # 5. Convert to node IDs - combine CoPe and embedding results
        start_node_ids = []
        initial_scores = {}
        
        # Add embedding-based results
        for mid in ranking_ids:
            try:
                node_id = self.storage.get_node_id_by_mid(int(mid))
                start_node_ids.append(node_id)
                initial_scores[node_id] = 1.0
            except (KeyError, IndexError):
                continue
        
        # Add CoPe search results
        concept_counts = self.storage.get_nodes_by_concepts(query_concepts) if query_concepts else {}
        for node_id in cope_node_ids:
            if node_id not in start_node_ids:
                start_node_ids.append(node_id)
            overlap_count = concept_counts.get(node_id, 0)
            initial_scores[node_id] = (overlap_count / len(query_concepts)) if query_concepts else 0.8
        
        if not start_node_ids:
            self.last_retrieved_ids = []
            return []
        
        # 6. CoPe Search Stage 2: Graph traversal through detailed contexts
        traversal_results = []
        if start_node_ids:
            traversal_results = self.__graph_traversal__(
                query_embedding,
                query_concepts,
                start_node_ids
            )
        
        # 7. Merge all candidates
        all_candidates = {}
        
        # Add initial scores (embedding + CoPe Stage 1)
        for node_id, score in initial_scores.items():
            all_candidates[node_id] = score
        
        # Add traversal results (CoPe Stage 2)
        for node_id, score in traversal_results:
            if node_id not in all_candidates:
                all_candidates[node_id] = 0.0
            all_candidates[node_id] = max(all_candidates[node_id], score)
        
        # 8. Sort and select top-k
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        selected_node_ids = [node_id for node_id, _ in sorted_candidates[:self.max_nodes]]
        
        if not selected_node_ids:
            self.last_retrieved_ids = []
            return []
        
        # 9. Collect memories
        memories = []
        retrieved_ids = []
        for node_id in selected_node_ids:
            mem = self.storage.get_memory_element_by_node_id(node_id)
            memories.append(mem)
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                retrieved_ids.append(mem['dialogue_id'])
        
        self.last_retrieved_ids = retrieved_ids
        
        # 10. Format using Utilization
        result = self.utilization(memories)
        return result


class UniversalRAGRecall(BaseRecall):
    """
    UniversalRAG Recall
    Supports 3 types: 'no', 'document', 'image'
    """
    def __init__(self, config, **kwargs):
        super().__init__(config)
        
        self.storage = kwargs['storage']
        self.routing = kwargs['routing']
        self.retrieval = kwargs['retrieval']
        self.truncation = eval(self.config.truncation.method)(self.config.truncation)
        self.utilization = eval(self.config.utilization.method)(self.config.utilization)
    
    def reset(self):
        self.__reset_objects__([self.truncation, self.utilization, self.routing, self.retrieval])
    
    @__recall_convert_str_to_observation__
    def __call__(self, query):
        if self.storage.is_empty():
            # Return empty list, consistent with MMMemoryRecall
            self.last_retrieved_ids = []
            return []

        # 1. Routing decision (returns 'no', 'document', or 'image')
        modality = self.routing.route(query)

        # 2. If 'no', return empty list directly
        if modality == 'no':
            self.last_retrieved_ids = []
            return []

        # 3. Retrieval
        retrieved_ids, scores = self.retrieval.retrieve(
            query, modality,
            top_k=getattr(self.config, 'top_k', 5)
        )

        # 4. Fallback mechanism: if retrieval results are empty or too few, and routed to 'image', try retrieving from 'document'
        if len(retrieved_ids) == 0 and modality == 'image':
            # Fallback to document retrieval
            retrieved_ids, scores = self.retrieval.retrieve(
                query, 'document',
                top_k=getattr(self.config, 'top_k', 5)
            )

        # 5. Get retrieved content
        retrieved_memories = []
        retrieved_ids_for_eval = []
        for mid in retrieved_ids:
            mem = self.storage.get_memory_element_by_mid(mid)
            retrieved_memories.append(mem)
            # Collect alignment IDs for evaluation
            if isinstance(mem, dict) and ('dialogue_id' in mem and mem['dialogue_id']):
                retrieved_ids_for_eval.append(mem['dialogue_id'])

        # Expose for evaluation
        self.last_retrieved_ids = retrieved_ids_for_eval

        # 6. Format and return (consistent with MMMemoryRecall)
        # MultiModalUtilization returns list, return directly
        result = self.utilization(retrieved_memories)
        return result

