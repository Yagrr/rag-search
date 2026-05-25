import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .utils_search import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    SCORE_PRECISION,
    DEFAULT_WEIGHTED_SEARCH_ALPHA,
    DEFAULT_RRF_SEARCH_K,
)

from .llm import (
    enhance_query,
)

from .rerank import (
    rerank_results
)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
# Search results are of format dict[id : score]
        self.idx = InvertedIndex() 
# Search results is a dictionary containing  data on [id, title, document, score, metadata]
        self.semantic_search = ChunkedSemanticSearch() 
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        if not os.path.exists(self.idx.path_index):
            self.idx.build()
            self.idx.save_cache()

    def _bm25_search(self, query, limit):
        self.idx.load_cache()
        return self.idx.search_bm25(query, limit)

    def weighted_search(self, query: str, alpha: float = DEFAULT_WEIGHTED_SEARCH_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[int, dict]:
        # bm25 ID indexed to 1 so we do -1 on all IDs
        res_bm25_raw  = self._bm25_search(query, limit * 500)
        res_bm25 = {id - 1 : score for id, score in res_bm25_raw.items()}
        res_semantic = self.semantic_search.search_chunks(query, limit * 500)

        scores_bm25 = list(res_bm25.values())
        scores_semantic = [doc["score"] for doc in res_semantic]

        # Normalize takes in a list as input and returns a list as output.
        scores_bm25_norm = normalize(scores_bm25)
        scores_semantic_norm = normalize(scores_semantic)

        
        # map values back to ID
        for id, bm25_norm in zip(res_bm25, scores_bm25_norm):
            res_bm25.update({id: bm25_norm})

        for i in range(len(res_semantic)):
            res_semantic[i].update({"score": scores_semantic_norm[i]})

        # map ID to (title, document, semantic, bm25, hybrid)
        res_hybrid = {}
        for id in res_bm25:
            title = self.documents[id]["title"]
            document = self.documents[id]["description"][:100]
            bm25 = res_bm25[id]
            res_hybrid.update(
                {
                    id: {
                        "title": title,
                        "document": document,
                        "bm25": round(bm25, SCORE_PRECISION),
                        "semantic": 0,
                    }
                }
            )

        for doc in res_semantic:
            id = doc["id"]
            title = self.documents[id]["title"]
            document = self.documents[id]["description"][:100]
            semantic = doc["score"]
            if id not in res_hybrid:
                res_hybrid.update({id: {
                    "title": title,
                    "document": document,
                    "bm25": 0
                }})

            res_hybrid[id].update({
                "semantic": round(semantic, SCORE_PRECISION)
            })

        for id in res_hybrid:
            hybrid = hybrid_score(res_hybrid[id]["bm25"], res_hybrid[id]["semantic"], alpha)
            res_hybrid[id].update({
                "hybrid": round(hybrid, SCORE_PRECISION)
            })

        # Sort by hybrid score descending, get results up to limit.
        res_hybrid_sorted = dict(
            sorted(
                res_hybrid.items(), key=lambda item: item[1].get("hybrid"), reverse=True
            )[:limit]
        )
        
        return res_hybrid_sorted

    def rrf_search(self, query, k: int = DEFAULT_RRF_SEARCH_K, limit: int = DEFAULT_SEARCH_LIMIT, rerank_method: str | None = None):
        res_bm25_raw = self._bm25_search(query, limit * 500) 
        res_bm25 = {id - 1 : score for id, score in res_bm25_raw.items()}
        res_semantic = self.semantic_search.search_chunks(query, limit * 500)

        scores_bm25 = list(res_bm25.values())
        scores_semantic = [doc["score"] for doc in res_semantic]

        # Normalize takes in a list as input and returns a list as output.
        scores_bm25_norm = normalize(scores_bm25)
        scores_semantic_norm = normalize(scores_semantic)

        
        # map values back to ID
        for id, bm25_norm in zip(res_bm25, scores_bm25_norm):
            res_bm25.update({id: bm25_norm})

        for i in range(len(res_semantic)):
            res_semantic[i].update({"score": scores_semantic_norm[i]})

        # map ID to (title, document, semantic, bm25, hybrid)
        res_hybrid = {}
        for i, id in enumerate(res_bm25, 1):
            title = self.documents[id]["title"]
            document = self.documents[id]["description"][:100]
            res_hybrid.update(
                {
                    id: {
                        "title": title,
                        "document": document,
                        "rank_bm25": i,
                        "rank_semantic": 0,
                    }
                }
            )

        for i, doc in enumerate(res_semantic, 1):
            id = doc["id"]
            title = self.documents[id]["title"]
            document_description = self.documents[id]["description"][:100]
            if id not in res_hybrid:
                res_hybrid.update(
                    {
                        id: {
                            "title": title,
                            "document": document_description,
                            "rank_bm25": 0,
                        }
                    }
                )

            res_hybrid[id].update({
                "rank_semantic": i
            })

        for id in res_hybrid:
            rank_bm25 = res_hybrid[id]["rank_bm25"]
            rank_semantic = res_hybrid[id]["rank_semantic"]
            res_hybrid[id].update(
                {
                    "rrf_score": rrf_score(rank_bm25, k) + rrf_score(rank_semantic, k),
                    "rerank_score": 0.0,
                    "rerank_rank": 0,
                    "rerank_crossencoder_score": 0.0,
                }
            )

        limit = int(limit / 5)

        # Sorting by rrf_score
        res_semantic_sorted = dict(
            sorted(
                res_hybrid.items(), key=lambda item: item[1].get("rrf_score"), reverse=True
            )[:limit]
        )

        # Rerank documents
        return rerank_results(query, res_semantic_sorted, rerank_method, limit)


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_score = min(values)
    max_score = max(values)

    if min_score == max_score:
        return [1.0] * len(values)

    norm_scores = []
    for s in values:
        norm_scores.append(
            (s - min_score) / (max_score - min_score)
        )
    return norm_scores

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = DEFAULT_WEIGHTED_SEARCH_ALPHA) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=DEFAULT_RRF_SEARCH_K) -> float:
    return (1 / (k + rank))

def command_weighted_search(query: str, alpha: float, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    documents = load_movies()
    search_instance = HybridSearch(documents)
    results = search_instance.weighted_search(query, alpha, limit)
    for i, res in enumerate(results.values()):
        print(f"{i+1}. {res["title"]}")
        print(f" Hybrid Score: {res["hybrid"]: .4f}")
        print(f" BM25: {res["bm25"]: .4f}, Semantic: {res["semantic"]: .4f}")
        print(f" {res["document"]}")

def command_rrf_search(
    query: str,
    k: int = DEFAULT_RRF_SEARCH_K,
    enhance_method: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_method: str | None = None,
) -> None:

    query_original = query
    if enhance_method is not None:
        query = enhance_query(query, enhance_method)
        print(f"Enhanced query ({enhance_method}): '{query_original}' -> '{query}'")

    match rerank_method:
        case "individual" | "batch" | "cross_encoder":
            limit = int(limit * 5)

    documents = load_movies()
    search_instance = HybridSearch(documents)
    results = search_instance.rrf_search(query, k, limit, rerank_method)

    for i, res in enumerate(results.values()):
        print(f"{i+1}. {res["title"]}")

        if res.get("rerank_score", 0) != 0:
            print(f" Re-rank Score: {res['rerank_score']}/10")
        if res.get("rerank_rank", 0) != 0:
            print(f" Re-rank Rank: {res['rerank_rank']}")
        if res.get("rerank_crossencoder_score", 0) != 0:
            print(f"Cross Encoder Score: {res['rerank_crossencoder_score']:.3f}")



        print(f" RRF Score: {res["rrf_score"]: .4f}")
        print(f" BM25 Rank: {res["rank_bm25"]}, Semantic Rank: {res["rank_semantic"]}")
        print(f" {res["document"]}")
