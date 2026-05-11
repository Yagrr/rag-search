import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.utils_search import (
    PATH_CACHE,
    load_movies,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_WORDS_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
)


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL)-> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.path_embeddings = os.path.join(PATH_CACHE, "movie_embeddings.npy")


    def generate_embedding(self, text: str):
        if text.isspace() or text == "":
            raise ValueError(f"Text is whitespace or is empty string '{text}'")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        docs_str = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            docs_str.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(docs_str, show_progress_bar=True)
        np.save(self.path_embeddings, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.path_embeddings):
            self.embeddings = np.load(self.path_embeddings)

            if len(self.embeddings) == len(documents):
                return self.embeddings

        self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedding_query = self.generate_embedding(query)
        cos_sim_query_doc = []

        for doc, doc_embedding in zip(self.documents,self.embeddings):
            cos_sim = cosine_similarity(embedding_query, doc_embedding)
            cos_sim_query_doc.append((cos_sim, doc))

        sorted_cos_sim = sorted(cos_sim_query_doc, key=lambda item: item[0], reverse=True)
        results = []
        for cos_sim in sorted_cos_sim:
            results.append(
                {
                    "score": cos_sim[0],
                    "title": cos_sim[1]["title"],
                    "description": cos_sim[1]["description"],
                }
            )
            if len(results) >= limit:
                break
        return results

def command_semantic_search(query: str, limit: int):
    search_instance = SemanticSearch()
    documents = list(load_movies())

    print("Loading/creating embeddings...")
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']})\n{res['description']}\n")


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity of two vectors, so that embeddings are not
    affected by vector magnitudes.
    """
    # Dot product
    dot_product = np.dot(vec1, vec2)
    # Magnitudes
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def verify_model(model_name):
    embedder = SemanticSearch(model_name)
    print(f"Model loaded: {embedder.model}")
    print(f"Max sequence length: {embedder.model.max_seq_length}")


def verify_embedding():
    embedder = SemanticSearch()
    documents = list(load_movies())
    embeddings = embedder.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text):
    embedder = SemanticSearch()
    embedding = embedder.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def chunking_fixed_size(text: str, chunk_size:int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_WORDS_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    words_count = len(words)
    i = 0
    while i < words_count:
        if overlap > 0 and i > 0:
            chunks_words = words[i - overlap : i + chunk_size + overlap]
            chunks.append(" ".join(chunks_words))
            i += chunk_size + overlap
        else:
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))
            i += chunk_size
    return chunks


def chunk_text(text: str, chunk_size:int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_WORDS_OVERLAP) -> None:
    chunks = chunking_fixed_size(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def chunking_semantic(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    while i < len(sentences):
        if overlap > 0 and i > 0:
            chunks_sentences = sentences[i - overlap : i + max_chunk_size - overlap]
            # check if needs further split
            print(chunks_sentences)
            print(i)
            chunks.append(" ".join(chunks_sentences))
            i += max_chunk_size - overlap
        else:
            chunks_sentences = sentences[i : i + max_chunk_size]
            chunks.append(" ".join(chunks_sentences))
            i += max_chunk_size
    return chunks

def chunk_text_semantically(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP) -> None:
    chunks = chunking_semantic(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
