from abc import ABC, abstractmethod
import torch

class ConceptBasedRetrieval:
    """
    Concept-based retrieval using semantic concept labels.
    Retrieves nodes based on concept overlap rather than embedding similarity.
    """
    
    def __init__(self, config):
        self.config = config
        self.storage = None  # Will be set to TagGraphStorage
        self.min_concept_overlap = getattr(config, 'min_concept_overlap', 1)
        self.concept_weight = getattr(config, 'concept_weight', 1.0)
    
    def reset(self):
        pass
    
    def set_storage(self, storage):
        """Set the TagGraphStorage instance."""
        self.storage = storage
    
    def retrieve_by_concepts(self, query_concepts, topk='config', with_score=False):
        """
        Retrieve nodes based on concept overlap.
        
        Args:
            query_concepts: list of concept strings
            topk: int or 'config' or False
            with_score: bool, whether to return scores
            
        Returns:
            If with_score=False: list of node_ids (as integers from tensorstore indices)
            If with_score=True: (scores, node_ids)
        """
        if self.storage is None:
            raise ValueError("Storage not set. Call set_storage() first.")
        
        if not query_concepts:
            if with_score:
                return torch.tensor([]), torch.tensor([])
            return torch.tensor([])
        
        # Normalize concepts
        query_concepts = [str(c).lower().strip() for c in query_concepts if c]
        
        if not query_concepts:
            if with_score:
                return torch.tensor([]), torch.tensor([])
            return torch.tensor([])
        
        # Get nodes by concepts with overlap counts
        concept_counts = self.storage.get_nodes_by_concepts(query_concepts)
        
        if not concept_counts:
            if with_score:
                return torch.tensor([]), torch.tensor([])
            return torch.tensor([])
        
        # Convert node_ids to mid (memory indices)
        node_scores = {}
        for node_id, overlap_count in concept_counts.items():
            if overlap_count >= self.min_concept_overlap:
                try:
                    mid = self.storage.get_mid_by_node_id(node_id)
                    # Score is based on overlap count (normalized by query length)
                    score = (overlap_count / len(query_concepts)) * self.concept_weight
                    node_scores[mid] = score
                except (KeyError, IndexError):
                    continue
        
        if not node_scores:
            if with_score:
                return torch.tensor([]), torch.tensor([])
            return torch.tensor([])
        
        # Sort by score
        sorted_items = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        mids = [mid for mid, _ in sorted_items]
        scores = [score for _, score in sorted_items]
        
        # Apply topk
        if topk == 'config':
            k = min(getattr(self.config, 'topk', 10), len(mids))
        elif isinstance(topk, int):
            k = min(topk, len(mids))
        elif topk is False:
            k = len(mids)
        else:
            k = len(mids)
        
        mids = mids[:k]
        scores = scores[:k]
        
        # Convert to tensors
        mids_tensor = torch.tensor(mids, dtype=torch.long)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        
        if with_score:
            return scores_tensor, mids_tensor
        else:
            return mids_tensor
    
    def retrieve_by_concept_graph(self, query_concepts, start_node_ids=None, max_depth=2):
        """
        Retrieve nodes using graph traversal based on concept associations.
        
        Args:
            query_concepts: list of concept strings
            start_node_ids: list of starting node IDs (optional)
            max_depth: int, maximum traversal depth
            
        Returns:
            list: (node_id, score) tuples
        """
        if self.storage is None:
            raise ValueError("Storage not set. Call set_storage() first.")
        
        if not query_concepts:
            return []
        
        query_concepts_set = set([str(c).lower().strip() for c in query_concepts if c])
        
        if not query_concepts_set:
            return []
        
        # If no start nodes provided, use direct concept matches
        if start_node_ids is None:
            concept_counts = self.storage.get_nodes_by_concepts(query_concepts)
            start_node_ids = list(concept_counts.keys())
        
        visited = set()
        candidates = {}
        
        def traverse(node_id, depth=0):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            
            # Calculate concept overlap score
            node_concepts = self.storage.get_concepts_by_node(node_id)
            overlap = query_concepts_set & node_concepts
            if overlap:
                score = len(overlap) / len(query_concepts_set)
                if node_id not in candidates or score > candidates[node_id]:
                    candidates[node_id] = score
            
            # Traverse neighbors connected by concept associations
            neighbors = self.storage.get_neighbors(node_id)
            for neighbor_id in neighbors:
                edge = self.storage.get_edges_from(node_id).get(neighbor_id, {})
                if edge.get('type') == 'concept_association':
                    traverse(neighbor_id, depth + 1)
        
        # Start traversal from all start nodes
        for node_id in start_node_ids:
            traverse(node_id, 0)
        
        # Convert to mid and return
        results = []
        for node_id, score in candidates.items():
            try:
                mid = self.storage.get_mid_by_node_id(node_id)
                results.append((mid, score))
            except (KeyError, IndexError):
                continue
        
        return sorted(results, key=lambda x: x[1], reverse=True)

