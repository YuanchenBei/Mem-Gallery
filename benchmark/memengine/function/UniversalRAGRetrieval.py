class UniversalRAGRetrieval:

    def __init__(self, config, **kwargs):
        self.config = config
        self.storage = kwargs.get('storage')  # UniversalRAGStorage
    
    def reset(self):
        pass
    
    def retrieve(self, query, modality, top_k=5):
        """
        Retrieve based on routing decision

        Args:
            query: Query (dict or str)
            modality: 'no', 'document', or 'image'
            top_k: Return top-k results

        Returns:
            (retrieved_mids, scores): List of retrieved mids and scores
        """
        return self.storage.retrieve_by_query(
            query=query,
            modality=modality,
            top_k=top_k
        )


















