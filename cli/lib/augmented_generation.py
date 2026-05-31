from .llm import call_llm, enhance_query
from .utils_search import load_movies, DEFAULT_SEARCH_LIMIT
from .hybrid_search import HybridSearch

def rag(query: str, docs: str):
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""
    return call_llm(prompt)

def search(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> dict:
    query_original = query

    documents: dict = load_movies()
    enhance_method = "rewrite"
    rerank_method = "cross_encoder"
    limit = 5
    k = 60

    if enhance_method is not None:
        query = enhance_query(query, enhance_method)
        print(f"Enhanced query ({enhance_method}): '{query_original}' -> '{query}'")

    if rerank_method in ["individual", "batch", "cross_encoder"]:
        limit = int(limit * 5)
    search_instance = HybridSearch(documents)
    return search_instance.rrf_search(query, k, limit, rerank_method)


def collate_results(search_results: dict) -> tuple[str, str]:
    """
    Return titles in format:
    - Title Example 1
    - Title Example 2

    Return docs for LLM parsing in format:
    1. Title: Example
        Description: Example description
    """
    search_results_titles = []
    search_results_docs = []

    for i, res in search_results.items():
        search_results_titles.append(f"- {res["title"]}")
        search_results_docs.append(f"""{i+1}. Title:{res["title"]}
        Description: {res["document"]}""")

    titles = "\n".join(search_results_titles)
    docs = "\n".join(search_results_docs)
    return titles, docs


def command_rag(query: str):
    search_results = search(query)
    titles, docs = collate_results(search_results)
    results_rag = rag(query, docs)
    print("Search Results:")
    print(titles)
    print(f"RAG Response:\n{results_rag}")

def summarize(query: str, results: str) -> str:
    prompt = f"""Provide information useful to the query below by synthesizing
    data from multiple search results in detail. The goal is to provide
    comprehensive information so that users know what their options are. Your
    response should be information-dense and concise, with several key pieces
    of information about the genre, plot, etc. of each movie. This should be
    tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return call_llm(prompt)

def command_summarize(query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> None:
    search_results = search(query, limit)
    titles, docs = collate_results(search_results)
    summary = summarize(query, docs)
    print("Search Results:")
    print(titles)
    print("LLM Summary:")
    print(summary)
