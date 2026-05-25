import time
import json
from random import randint

from sentence_transformers import CrossEncoder

from .llm import call_llm
from .utils_search import (
    DEFAULT_MODEL_PROMPTING_SECONDS_DELAY,
    DEFAULT_CROSS_ENCODER
)

def rerank_results(query: str, results: dict, rerank_method: str | None, limit: int) -> dict:
    print(f"Re-ranking documents ({rerank_method})")
    match rerank_method:
        case "individual":
            """
                Re-rank documents using LLM.
            """
            for id in results:
                # Update score
                results[id]["rerank_score"] = rerank_individual(query, results[id])
                time.sleep(DEFAULT_MODEL_PROMPTING_SECONDS_DELAY * randint(1,3))
                print("...")

            results_reranked_sorted = dict(
                sorted(
                    results.items(),
                    key=lambda item: item[1].get("rerank_score"),
                    reverse=True,
                )[:limit]
            )
            return results_reranked_sorted
        
        case "batch":
            """
                Re-rank documents in batches using LLM.
                Batch processing allows for documents to be scored in the
                context of each other.
            """
            # Get string of doc list, ID, title, summary
            doc_list_str = []
            for id in results:
                doc_list_str.extend(["ID:", id, "\n", "title:", results[id]["title"], "description:", results[id]["document"], "\n"])

            doc_list_str = " ".join([str(doc) for doc in doc_list_str])

            reranked_ids = json.loads(rerank_batch(query, doc_list_str))
            print("...")

            for rank, id in enumerate(reranked_ids, 1):
                results[id]["rerank_rank"] = rank

            results_reranked_sorted = dict(
                sorted(
                    results.items(),
                    key=lambda item: item[1].get("rerank_rank")
                )[:limit]
            )

            return results_reranked_sorted

        case "cross_encoder":
            pairs = []
            for id in results:
                doc = results[id]
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            scores = rerank_crossencoder(pairs)

            for i, id in enumerate(results):
                results[id]["rerank_crossencoder_score"] = scores[i]

            results_reranked_sorted = dict(
                sorted(
                    results.items(),
                    key=lambda item: item[1].get("rerank_crossencoder_score"),
                    reverse=True,
                )[:limit]
            )
            return results_reranked_sorted

        case _:
            return results

def rerank_individual(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.
        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "").strip().strip('"')}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:
        """

    print(f"Reranking {doc.get("title", "")} - score:")
    try:
        score = call_llm(prompt)
        score = score.strip().strip('"')
    except Exception as e:
        print(f"Error - Retrying query: {e}")
        score = rerank_individual(query, doc)
    score = float(score)
    print(score)
    return score

def rerank_batch(query: str, doc_list_str: str) -> str:
    prompt = f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return the movie IDs in order of relevance, best match first.

        Your response must be a raw JSON array of integers.
        Do not wrap the JSON in Markdown. Do not use a ```json code block.
        Do not include any explanatory text.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:
        """
    result_json = call_llm(prompt)
    return result_json

def rerank_crossencoder(pairs: list[list[str]]):
    cross_encoder = CrossEncoder(DEFAULT_CROSS_ENCODER)
    try:
        scores = cross_encoder.predict(pairs)
    except ValueError as e:
        print(f"{e} - Error while using cross encoder")
        return []
    return scores
