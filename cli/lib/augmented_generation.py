from re import search
from .llm import call_llm, enhance_query
from .utils_search import load_movies
from .hybrid_search import HybridSearch

def rag(query: str, docs: str):
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""
    results = call_llm(prompt)
    return results

def command_rag(query: str):

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
    search_results: dict = search_instance.rrf_search(query, k, limit, rerank_method)

    search_results_titles = []
    search_results_docs = []

    for i, res in search_results.items():
        search_results_titles.append(res["title"])
        search_results_docs.append(f"""{i+1}. Title:{res["title"]}
        Description: {res["document"]}""")

    docs = "\n".join(search_results_docs)
    results_rag = rag(query, docs)
    print("Search Results:")
    for title in search_results_titles:
        print(f"- {title}")
    print(f"RAG Response:\n{results_rag}")
