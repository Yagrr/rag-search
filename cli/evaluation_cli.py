import argparse
import json

from lib.utils_search import load_movies, PATH_EVALUATION_DATASET
from lib.hybrid_search import HybridSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
    type=int,
    default=5,
    help="Number of results to evaluate (k for precision@k, recall@k)",
)

    args = parser.parse_args()
    limit = args.limit

    with open(PATH_EVALUATION_DATASET, "r") as file:
        data_evaluation = json.load(file)

    documents = load_movies()

    for test_case in data_evaluation["test_cases"]:
        query = test_case["query"]
        k = 60
        rerank_method = None
        search_instance = HybridSearch(documents)
        search_results: dict = search_instance.rrf_search(query, k, limit, rerank_method)
        titles_retrieved: list[str] = []
        titles_retrieved_relevant: list[str] = []
        for id in search_results:
            title = search_results[id].get("title", "")
            if title in test_case["relevant_docs"]:
                titles_retrieved_relevant.append(title)
            titles_retrieved.append(title)

        if titles_retrieved: 
            # precision = relevant_retrieved / total_retrieved
            precision_score: float = len(titles_retrieved_relevant) / len(titles_retrieved)
            # recall = relevant_retrieved / total_relevant
            total_relevant: int = len(test_case["relevant_docs"])
            recall_score: float = len(titles_retrieved_relevant) / total_relevant
        else:
            precision_score = 0.0
            recall_score = 0.0

        if (precision_score + recall_score) == 0:
            f1_score: float = 0.0
        else:
            f1_score: float = 2 * (precision_score * recall_score) / (precision_score +  recall_score)

        print(f"k={limit}\n")
        print(f"- Query: {query}")
        print(f" - Precision@{limit}: {precision_score:.4f}")
        print(f" - Recall@{limit}: {recall_score:.4f}")
        print(f" - F1 Score: {f1_score:.4f}")
        print(f" - Retrieved: {', '.join(titles_retrieved)}")
        print(f" - Relevant: {', '.join(titles_retrieved_relevant)}")


if __name__ == "__main__":
    main()
