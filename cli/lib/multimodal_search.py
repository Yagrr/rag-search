import os

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.utils_search import DEFAULT_SEARCH_LIMIT, load_movies, PATH_CACHE, PROJECT_ROOT
from lib.semantic_search import cosine_similarity

class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str ="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.document_map = {}
        self.texts: list[str] = self.load_texts(documents)
        self.text_embeddings = None
        self.path_text_embeddings = os.path.join(PATH_CACHE, "multimodal_text_embeddings.npy")

    def load_texts(self, documents: list[dict]) -> list[str]:
        texts = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            texts.append(f"{doc["title"]}: {doc["description"]}")
        return texts

    def embed_image(self, path_image: str):
        if not os.path.exists(path_image):
            raise ValueError(f"Error - Path to image does not exist: {path_image}")

        image = Image.open(path_image)
        return self.model.encode([image])[0]

    def build_text_embeddings(self): 
        os.makedirs(PATH_CACHE, exist_ok=True)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
        np.save(self.path_text_embeddings, self.text_embeddings)
        return self.text_embeddings

    def load_or_create_text_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.path_text_embeddings):
            self.text_embeddings = np.load(self.path_text_embeddings)
            if len(self.text_embeddings) == len(documents):
                return self.text_embeddings
        
        return self.build_text_embeddings()

    def search_with_image(self, path_image: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[dict]:
        if not os.path.exists(path_image):
            print(f"Image does not exist at path: {path_image}") 
            return []

        if self.text_embeddings is None:
            raise ValueError("No text embeddings loaded. Call `load_or_create_text_embeddings()` first.")

        embedding_image = self.embed_image(path_image)
        scores_to_doc: list[tuple[float, dict]] = []

        for doc, embedding_text in zip(self.documents, self.text_embeddings):
            # Map text embeddings back to the actual documents
            cosine_score = cosine_similarity(embedding_image, embedding_text)
            scores_to_doc.append((cosine_score, doc))

        scores_to_doc_sorted = sorted(scores_to_doc, key=lambda item: item[0], reverse=True)

        results: list[dict] = []
        for score_doc in scores_to_doc_sorted:
            score = score_doc[0]
            doc = score_doc[1]
            results.append(
                {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "description": doc.get("description"),
                    "score": score,

                }
            )
            if len(results) >= limit:
                break
        return results
        

def verify_image_embedding(path_image: str) -> None:
    if not os.path.exists(path_image):
        print(f"Image does not exist at path: {path_image}") 
        return

    documents = list(load_movies())

    embedder = MultimodalSearch(documents)
    embedding = embedder.embed_image(path_image)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def command_image_search(path_image: str, limit: int=DEFAULT_SEARCH_LIMIT) -> None:
    documents = list(load_movies())
    search_instance = MultimodalSearch(documents)
    path_image = os.path.join(PROJECT_ROOT, path_image)

    print("Loading/creating embeddings...")
    search_instance.load_or_create_text_embeddings(documents)
    results = search_instance.search_with_image(path_image, limit)

    for i, res in enumerate(results):
        print(f"{i+1} {res["title"]} (similarity: {res["score"]:.4f})")
        print(f"{res["description"][:100]}...")
