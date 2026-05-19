import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.path_index):
            self.idx.build()
            self.idx.save_cache()

    def _bm25_search(self, query, limit):
        self.idx.load_cache()
        return self.idx.search_bm25(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def command_normalize(values: list[float]) -> None:
    if not values:
        return

    min_score = min(values)
    max_score = max(values)

    if min_score == max_score:
        scores = ([1] * len(values))
    else:
        scores = [((s - min_score ) / (max_score - min_score)) for s in values ]
    
    for score in scores:
        print(f"* {score:.4f}")

