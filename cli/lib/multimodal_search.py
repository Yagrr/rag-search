import os

from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def embed_image(self, path_image: str):
        if not os.path.exists(path_image):
            raise ValueError(f"Error - Path to image does not exist: {path_image}")

        image = Image.open(path_image)
        return self.model.encode([image])[0]


def verify_image_embedding(path_image: str) -> None:
    if not os.path.exists(path_image):
        print(f"Image does not exist at path: {path_image}") 
        return
    embedder = MultimodalSearch()
    embedding = embedder.embed_image(path_image)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def command_multimodal_search(path_image: str) -> None:
    verify_image_embedding(path_image)
