import os
import numpy as np
from sentence_transformers import SentenceTransformer

from lib.utils_search import PATH_CACHE, load_movies

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
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

        self.embeddings = self.model.encode(docs_str, show_progress_bar = True)
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


def verify_model(model_name):
    embedder = SemanticSearch(model_name)
    print(f"Model loaded: {embedder.model}")
    print(f"Max sequence length: {embedder.model.max_seq_length}")


def verify_embedding():
    embedder = SemanticSearch()
    documents = list(load_movies())
    embeddings = embedder.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


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
