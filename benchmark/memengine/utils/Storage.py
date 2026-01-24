from abc import ABC, abstractmethod
from memengine.function.Truncation import *

class BaseStorage(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass

class LinearStorage(BaseStorage):
    """
    Memory storage in linear structure.
    """
    def __init__(self, config):
        super().__init__(config)

        self.memory_list = []
        self.counter = 0

    def reset(self):
        self.clear_memory()
        self.counter = 0

    def display(self):
        memory_display_items = []
        for m in self.memory_list:
            memory_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in m.items()]))
        if len(memory_display_items) == 0:
            return 'None'
        return '\n'.join(['[Memory Entity %d]\n%s' % (index, m) for index, m in enumerate(memory_display_items)])

    def clear_memory(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.get_element_number()
        
        assert 0 <= start <= end <= self.get_element_number()

        if start == 0:
            if end == self.get_element_number():
                self.memory_list = []
            else:
                self.memory_list = self.memory_list[end:]
        else:
            if end == self.get_element_number():
                self.memory_list = self.memory_list[:start]
            else:
                self.memory_list = self.memory_list[0:start] + self.memory_list[end:]

    def get_element_number(self):
        return len(self.memory_list)

    def is_empty(self):
        return self.get_element_number() == 0
    
    def get_memory_element_by_mid(self, mid):
        return self.memory_list[mid]

    def get_memory_attribute_by_mid(self, mid, attr):
        return self.memory_list[mid][attr]

    def get_memory_text_by_mid(self, mid):
        return self.get_memory_attribute_by_mid(mid, 'text')
    
    def get_memory_image_by_mid(self, mid):
        """
        Get image information of a specified memory
        """
        return self.get_memory_attribute_by_mid(mid, 'image')
    
    def get_memory_timestamp_by_mid(self, mid):
        """Get the timestamp of the specified memory"""
        return self.get_memory_attribute_by_mid(mid, 'timestamp')

    def get_memory_dialogue_id_by_mid(self, mid):
        """Get the dialogue_id of the specified memory"""
        return self.get_memory_attribute_by_mid(mid, 'dialogue_id')

    def get_mids_by_attribute(self, attr, value):
        mids = []
        for mid, m in enumerate(self.memory_list):
            if attr in m and m[attr] == value:
                mids.append(mid)
        return mids
    
    def update_memory_attribute_by_mid(self, mid, attr, value):
        self.memory_list[mid][attr] = value
    
    def add(self, obj):
        """
        Add memory objects to support new multimodal data formats.
        Memory Record (initial): 
        {"text": ["msg1", "msg2", ...],
        "image": {"path"=["path1", "path2", ...], 
                  "caption": ["caption1", "caption2", ...],
                  "img_id": ["img_id 1", "img id 2", ...]},
        "timestamp": xxx,
        "dialogue_id": [xxx],
        counter_id: xxx
        }

        Memory Record (new): 
        {"text (str)": "msg",
        "image (dict)": {"path": "xxx", 
                  "caption": "xxx",
                  "img_id": "xxx"},
        "timestamp (str)": xxx,
        "dialogue_id (str)": xxx,
        counter_id (int): xxx
        }
        """
        if 'text' not in obj and 'image' not in obj:
            raise ValueError("Memory object must contain at least 'text' or 'image' field")
        
        obj['counter_id'] = self.counter
        self.memory_list.append(obj)
        self.counter += 1

    def delete_by_mid(self, mid):
        self.memory_list.pop(mid)

    def delete_by_mid_list(self, mids):
        for mid in sorted(mids, reverse=True):
            self.delete_by_mid(mid)

    def get_all_memory_in_order(self):
        return self.memory_list



class GraphStorage(BaseStorage):
    """
    Memory storage in graph structure.
    """
    def __init__(self, config):
        super().__init__(config)

        self.node = {}
        self.edge = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []

    def reset(self):
        self.node = {}
        self.edge = {}
        self.node_counter = 0
        self.edge_counter = 0
        self.memory_order_map = []

    def display(self):
        node_display_items = []
        for node_id, element in self.node.items():
            node_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in element.items()]))
        if len(node_display_items) == 0:
            return 'None'
        
        edge_display_items = []
        for edge_id, element in self.edge.items():
            edge_display_items.append('\n'.join(['%s: %s' % (k,v) for k, v in element.items()]))

        if len(node_display_items) == 0:
            node_context =  'None'
        else:
            node_context = '\n'.join(['[Node Entity %d]\n%s' % (index, m) for index, m in enumerate(node_display_items)])
        
        if len(edge_display_items) == 0:
            edge_context =  'None'
        else:
            edge_context = '\n'.join(['[Edge Entity %d]\n%s' % (index, m) for index, m in enumerate(edge_display_items)])
        return """Memory Node Context::
%s
Memory Edge Context::
%s""" % (node_context, edge_context)
    
    def get_element_number(self):
        return len(self.node)

    def is_empty(self):
        return self.get_element_number() == 0
    
    def get_node_id_by_mid(self, mid):
        return self.memory_order_map[mid]

    def get_mid_by_node_id(self, node_id):
        return self.node[node_id]['mid']

    def get_memory_element_by_node_id(self, node_id):
        return self.node[node_id]

    def get_memory_element_by_mid(self, mid):
        node_id = self.get_node_id_by_mid(mid)
        return self.node[node_id]

    def get_memory_text_by_node_id(self, node_id):
        return self.node[node_id].get('text', '')

    def get_memory_text_by_mid(self, mid):
        return self.get_memory_element_by_mid(mid).get('text', '')
    
    def get_memory_image_by_node_id(self, node_id):
        """Get the image information of a specified node"""
        return self.node[node_id].get('image', None)
    
    def get_memory_image_by_mid(self, mid):
        """Get image information of a specified memory"""
        return self.get_memory_element_by_mid(mid).get('image', None)
    
    def get_neighbors(self, node_id):
        """Get the list of neighboring node IDs of a node"""
        if node_id in self.edge:
            return list(self.edge[node_id].keys())
        return []
    
    def get_edges_from(self, node_id):
        """Get all edges originating from the specified node"""
        if node_id in self.edge:
            return self.edge[node_id]
        return {}
    
    def get_edges_to(self, node_id):
        """Get all edges pointing to a specified node"""
        edges_to = {}
        for source, targets in self.edge.items():
            if node_id in targets:
                edges_to[source] = targets[node_id]
        return edges_to
    
    def get_node_degree(self, node_id):
        """
        Matching official implementation: self.graph.degree(node_id)
        """
        out_degree = len(self.get_neighbors(node_id))
        
        in_degree = len(self.get_edges_to(node_id))
        
        return out_degree + in_degree

    def update_memory_attribute_by_node_id(self, node_id, attr, value):
        self.node[node_id][attr] = value
    
    def update_memory_attribute_by_mid(self, mid, attr, value):
        node_id = self.get_node_id_by_mid(mid)
        self.node[node_id][attr] = value

    def __update_memory_order_map__(self):
        node_id_list = list(self.node.keys())
        self.memory_order_map = node_id_list
        for mid, node_id in enumerate(node_id_list):
            self.node[node_id]['mid'] = mid

    def add_node(self, obj):
        # Support both text-only and multimodal nodes
        if 'text' not in obj and 'image' not in obj:
            raise ValueError("Node object must contain at least 'text' or 'image' field")
        obj['node_id'] = self.node_counter
        obj['mid'] = len(self.memory_order_map)
        self.node[self.node_counter] = obj
        self.memory_order_map.append(self.node_counter)

        self.node_counter += 1
        return self.node_counter - 1

    def add_edge(self, s, t, obj):
        obj['edge_id'] = self.edge_counter
        if s not in self.edge:
            self.edge[s] = {}
        if t not in self.edge[s]:
            self.edge[s][t] = obj
        
        self.edge_counter += 1
        return self.edge_counter - 1


class TagGraphStorage(GraphStorage):
    """
    Graph storage with concept tag support.
    Extends GraphStorage to support semantic concept labels and concept-based retrieval.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # Concept index: concept -> list of node_ids
        self.concept_index = {}
        
        # Node concept mapping: node_id -> set of concepts
        self.node_concepts = {}
    
    def reset(self):
        super().reset()
        self.concept_index = {}
        self.node_concepts = {}
    
    def add_node(self, obj):
        """
        Add node with concept tags support.
        """
        # Extract concepts if present
        concepts = obj.get('concepts', [])
        if not isinstance(concepts, list):
            concepts = []
        
        # Normalize concepts
        concepts = [str(c).lower().strip() for c in concepts if c]
        
        # Add node using parent class
        node_id = super().add_node(obj)
        
        # Store concepts for this node
        self.node_concepts[node_id] = set(concepts)
        
        # Update concept index
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            if node_id not in self.concept_index[concept]:
                self.concept_index[concept].append(node_id)
        
        return node_id
    
    def get_nodes_by_concept(self, concept):
        """
        Get all node IDs that contain a given concept.
        
        Args:
            concept: str, concept label
            
        Returns:
            list: List of node IDs
        """
        concept = str(concept).lower().strip()
        return self.concept_index.get(concept, [])
    
    def get_nodes_by_concepts(self, concepts):
        """
        Get all node IDs that contain any of the given concepts.
        
        Args:
            concepts: list of concept strings
            
        Returns:
            dict: {node_id: overlap_count} mapping
        """
        concept_counts = {}
        
        for concept in concepts:
            concept = str(concept).lower().strip()
            node_ids = self.concept_index.get(concept, [])
            for node_id in node_ids:
                concept_counts[node_id] = concept_counts.get(node_id, 0) + 1
        
        return concept_counts
    
    def get_concepts_by_node(self, node_id):
        """
        Get all concepts associated with a node.
        
        Args:
            node_id: int, node ID
            
        Returns:
            set: Set of concept strings
        """
        return self.node_concepts.get(node_id, set())
    
    def update_node_concepts(self, node_id, concepts):
        """
        Update concepts for an existing node.
        
        Args:
            node_id: int, node ID
            concepts: list of concept strings
        """
        # Remove old concepts from index
        old_concepts = self.node_concepts.get(node_id, set())
        for concept in old_concepts:
            if concept in self.concept_index and node_id in self.concept_index[concept]:
                self.concept_index[concept].remove(node_id)
                if not self.concept_index[concept]:
                    del self.concept_index[concept]
        
        # Normalize new concepts
        concepts = [str(c).lower().strip() for c in concepts if c]
        
        # Update node concepts
        self.node_concepts[node_id] = set(concepts)
        
        # Update concept index
        for concept in concepts:
            if concept not in self.concept_index:
                self.concept_index[concept] = []
            if node_id not in self.concept_index[concept]:
                self.concept_index[concept].append(node_id)
        
        # Update node object
        if node_id in self.node:
            self.node[node_id]['concepts'] = concepts
    
    def add_concept_edge(self, source_node_id, target_node_id, shared_concepts):
        """
        Add an edge based on shared concepts.
        
        Args:
            source_node_id: int, source node ID
            target_node_id: int, target node ID
            shared_concepts: list of shared concept strings
        """
        self.add_edge(
            source_node_id,
            target_node_id,
            {
                'type': 'concept_association',
                'shared_concepts': shared_concepts,
                'weight': len(shared_concepts)
            }
        )
    
    def get_concept_statistics(self):
        """
        Get statistics about concepts in the graph.
        
        Returns:
            dict: Statistics including total concepts, most common concepts, etc.
        """
        concept_counts = {concept: len(node_ids) for concept, node_ids in self.concept_index.items()}
        
        return {
            'total_concepts': len(self.concept_index),
            'total_nodes_with_concepts': len(self.node_concepts),
            'concept_counts': concept_counts,
            'most_common_concepts': sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
