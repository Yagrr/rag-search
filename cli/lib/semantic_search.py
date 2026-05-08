from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if text.isspace() or text == "":
            raise ValueError(f"Text is whitespace or is empty string '{text}'")
        embedding = self.model.encode(list(text))
        return embedding[0]


def verify_model(model_name):
    embedder = SemanticSearch(model_name)
    print(f"Model loaded: {embedder.model}")
    print(f"Max sequence length: {embedder.model.max_seq_length}")


def embed_text(text):
    embedder = SemanticSearch()
    embedding = embedder.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
