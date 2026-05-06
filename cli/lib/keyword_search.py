import os
import pickle
import string

from nltk.stem import PorterStemmer

from .utils_search import (
    PATH_CACHE, 
    DEFAULT_SEARCH_LIMIT, 
    load_stopwords, 
    load_movies
)


# ======== Inverted Index ========
class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, str] = {}
        self.path_index = os.path.join(PATH_CACHE, "index.pkl")
        self.path_docmap  = os.path.join(PATH_CACHE, "docmap.pkl")


    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenize the input text, then add each token to the index with the document ID
        """
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        return

    def get_documents(self, term: str) -> list[int]:
        """
        Get the set of document IDs for a given token, and return them as a
        list, sorted in ascending order.
        Assuming that the input term is a single word or token.
        """
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self) -> None:
        """
        Iterate over all indexes with `__add_document()`, concatenate both
        the title and description for use as input text.
        """
        # TODO: Hard coded data sources and field for now. Need to create more flexible API
        # Future implementation: pass input list[str] corresponding to fields
        # add for loop that iterates over datum field and appends to datum_tokens
        data = load_movies()
        field_1 = "title"
        field_2 = "description"

        for datum in data:
            doc_id = datum["id"]
            datum_field_info = f"{datum[field_1]} {datum[field_2]}"
            self.docmap[doc_id] = datum
            self.__add_document(doc_id, datum_field_info)
            print(f"Added: {id}. {datum[field_1]}")

        return

    def save_cache(self) -> None:
        """
        Save the index and docmap attributes to disk using pickle module's dump
        function. Destination path defaults to /cache/.
        """
        os.makedirs(PATH_CACHE, exist_ok=True)

        try:
            with open(self.path_index, "wb") as file:
                pickle.dump(self.index, file)
                print(f"Save index.pkl to disk at : {self.path_index}")

            with open(self.path_docmap, "wb") as file:
                pickle.dump(self.docmap, file)
                print(f"Save docmap.pkl to disk at : {self.path_docmap}/")
        except Exception as e:
            print(f"Error while saving: {e}")

    def load_cache(self) -> None: 
        """ 
        Loads index.pkl and docmap.pkl from path_dst folder then sets the
        index and docmap of the InvertedIndex to the loaded data.
        """

        try:
            with open(self.path_index, "rb") as file_index:
                index = pickle.load(file_index)
            with open(self.path_docmap, "rb") as file_docmap:
                docmap = pickle.load(file_docmap)

            self.index = index
            self.docmap = docmap

        except Exception as e:
            print(f"Error while loading: {e}")

# ======== Pre-processing  ========


def command_search(query: str, field_to_search: str = "title", limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    matches = []
    matching_ids = []

    index = InvertedIndex()

    try:
        index.load_cache()
    except Exception as e:
        print(f"Error loading index from cache while searching: {e}")

    tokens_query: list[str] = stem_text(tokenize_text(preprocess_text(query)))

    for tk in tokens_query:
        matching_ids.extend(index.get_documents(tk))

    for id in matching_ids:
        matches.append(index.docmap[id])
    return matches


def preprocess_text(input_str: str) -> str:
    """Cull non alphanumeric characters"""
    input_str = input_str.lower()
    input_str = input_str.translate(str.maketrans("", "", string.punctuation))

    return input_str


def tokenize_text(input_str: str) -> list[str]:
    """Tokenize text by whitespace inbetween words.
    Cull empty strings
    Cull stop words
    """
    stop_words = load_stopwords()
    tokens = [s.lower() for s in input_str.split() if s != "" and s not in stop_words]
    return tokens


def stem_text(tokens_list: list[str]) -> list[str]:
    tokens_stemmed = []

    stemmer = PorterStemmer()

    for token in tokens_list:
        tokens_stemmed.append(stemmer.stem(token))

    return tokens_stemmed
