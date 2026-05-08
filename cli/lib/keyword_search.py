import os
import math
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .utils_search import (
    BM25_K1,
    BM25_B,
    PATH_CACHE, 
    DEFAULT_SEARCH_LIMIT, 
    load_stopwords, 
    load_movies
)


# ======== Inverted Index ========
class InvertedIndex:
    def __init__(self) -> None:
        # Index of token to document object
        self.index = defaultdict(set)

        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths: dict = {}

        self.path_index = os.path.join(PATH_CACHE, "index.pkl")
        self.path_docmap = os.path.join(PATH_CACHE, "docmap.pkl")
        self.path_term_frequencies = os.path.join(PATH_CACHE, "term_frequencies.pkl")
        self.path_doc_lengths = os.path.join(PATH_CACHE, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenize the input text, then add each token to the index with the document ID
        """
        tokens = tokenize_text(text)

        total_tokens = len(tokens)

        self.doc_lengths[doc_id] = total_tokens

        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
        for token in set(tokens):
            self.index[token].add(doc_id)
        return


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

        for doc in data:
            doc_id = doc["id"]
            doc_info = f"{doc[field_1]} {doc[field_2]}"
            self.docmap[doc_id] = doc
            self.__add_document(doc_id, doc_info)
            print(f"Added: {doc_id}. {doc[field_1]}")
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
            with open(self.path_docmap, "wb") as file:
                pickle.dump(self.docmap, file)
            with open(self.path_term_frequencies, "wb") as file:
                pickle.dump(self.term_frequencies, file)
            with open(self.path_doc_lengths, "wb") as file:
                pickle.dump(self.doc_lengths, file)
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
            with open(self.path_term_frequencies, "rb") as file_tf:
                term_frequencies = pickle.load(file_tf)
            with open(self.path_doc_lengths, "rb") as file_doc_lengths:
                doc_lengths = pickle.load(file_doc_lengths)

            self.index = index
            self.docmap = docmap
            self.term_frequencies = term_frequencies
            self.doc_lengths = doc_lengths

        except Exception as e:
            print(f"Error while loading: {e}")

    def get_documents(self, term: str) -> list[int]:
        """
        Get the set of document IDs for a given token, and return them as a
        list, sorted in ascending order.
        Assuming that the input term is a single word or token.
        """
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Error - term must be single token '{term}'")

        doc_ids = self.index.get(tokens[0], set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Get term frequencies for a given term and document_id in term_frequencies attribute.

        Passes through term to tokenize_text() for preprocessing, and stemming.
        Raise value error if more than one token.
        """
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError(f"Error - term must be single token: '{term}'")

        return self.term_frequencies[doc_id][tokens[0]]

    def get_idf(self, term: str) -> float:
        """
        Get the basic inverted document frequency (IDF) score for a given term.
        """

        count_total_docs = len(self.docmap)
        count_docs_with_term = len(self.get_documents(term))
        idf =  math.log((count_total_docs + 1) / (count_docs_with_term + 1))
        return idf

    def get_tfidf(self, doc_id: int, term: str) -> float:
        """
        Get the TF-IDF score for a given document and term.
        Calculated with `TF-ID = TF * IDF`
        """
        tfidf = self.get_tf(doc_id, term) * self.get_idf(term)
        return tfidf

    def __get_avg_doc_length(self) -> float:

        length_all_documents = sum(self.doc_lengths.values())
        number_of_documents = len(self.doc_lengths)

        if number_of_documents == 0:
            return 0.0
        else:
            return length_all_documents / number_of_documents

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """
        Get BM25 saturated term frequency for a given term and doc_id. Serves
        as an updated version from standard TF to avoid high scoring words that
        appear frequently.

        Passes through term to tokenize_text() for preprocessing, and stemming.
        Raise value error if more than one token.
        """
        tokens = tokenize_text(term)

        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        if len(tokens) != 1:
            raise ValueError(f"Error - term must be single token: '{term}'")

        tf = self.term_frequencies[doc_id][tokens[0]]
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str) -> float:
        """
        Get the inverted document frequency (BM25 IDF) score for a given term using
        the Okapi BM25 algorithm.
        """
        count_total_docs = len(self.docmap)
        count_docs_with_term = len(self.get_documents(term))
        laplace_smoothing = 0.5
        bm25_idf = math.log(
            (count_total_docs - count_docs_with_term + laplace_smoothing)
            / (count_docs_with_term + laplace_smoothing) 
            + 1
        )
        return bm25_idf
    
    def get_bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def search_bm25(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict[int, float]:
        """
        Returns doc_id and its BM25 score given an input query.
        
        Tokenizes query, calculates the BM25 score for each token for each
        indexed doc_id and appends its score and ID to a scores dictionary.
        Sort scores dictionary to find the top most relevant document, then
        return the top `limit` documents along with their scores.
        """
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        results = []

        for doc_id in self.docmap.keys():
            doc_id_total_bm25 = 0
            for token in tokens:
                bm25 = self.get_bm25(doc_id, token)
                doc_id_total_bm25 += bm25
            scores[doc_id] = doc_id_total_bm25

        scores_sorted_desc = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

        for score in scores_sorted_desc.items():
            results.append(score)
            if len(results) >= limit:
                break

        return dict(results)


# ======== Main command  ========

def create_InvertedIndex() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load_cache()
    except Exception as e:
        print(f"Error loading index from cache while searching: {e}")
    return index


def command_build() -> None:
    index = InvertedIndex()
    index.build()
    index.save_cache()
    print(f"Save index.pkl to disk at : {index.path_index}")
    print(f"Save docmap.pkl to disk at : {index.path_docmap}")
    print(f"Save term_frequencies.pkl to disk at : {index.path_term_frequencies}")
    print(f"Save path_lenghts.pkl to disk at : {index.path_doc_lengths}")
    

def command_search(query: str, field_to_search: str = "title", limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """
    Command for querying words in cached inverted index data.
    Takes a string query, and an integer limit for how many results should be fetched.
    Returns a list of dictionary document objects.
    """
    matches = []
    doc_ids = []
    index = create_InvertedIndex()

    tokens_query: list[str] = tokenize_text(query)

    for tk in tokens_query:
        doc_ids.extend(index.get_documents(tk))

    for id in doc_ids:
        matches.append(index.docmap[id])
        if len(matches) >= limit:
            break

    return matches


def command_tf(doc_id: int, term: str) -> int:
    index = create_InvertedIndex()
    return index.get_tf(doc_id, term)


def command_idf(term: str) -> float:
    index = create_InvertedIndex()
    return index.get_idf(term)


def command_tfidf(doc_id: int, term: str):
    index = create_InvertedIndex()
    return index.get_tfidf(doc_id, term)

def command_bm25_tf(doc_id: int, term: str, k1: float) -> float:
    index = create_InvertedIndex()
    return index.get_bm25_tf(doc_id, term, k1)

def command_bm25_idf(term: str) -> float:
    index = create_InvertedIndex()
    return index.get_bm25_idf(term)

def command_bm25_search(query: str, limit: int) -> tuple[list[dict],  dict[int, float]]:
    matches = []
    index = create_InvertedIndex()
    scores = index.search_bm25(query, limit) 

    for id in scores.keys():
        matches.append(index.docmap[id])

    return matches, scores

# ======== Pre-processing  ========


def preprocess_text(input_str: str) -> str:
    """Cull non alphanumeric characters"""
    input_str = input_str.lower()
    input_str = input_str.translate(str.maketrans("", "", string.punctuation))

    return input_str


def stem_tokens(tokens_list: list[str]) -> list[str]:
    tokens_stemmed = []

    stemmer = PorterStemmer()

    for token in tokens_list:
        tokens_stemmed.append(stemmer.stem(token))

    return tokens_stemmed


def tokenize_text(input_str: str) -> list[str]:
    """Tokenize text by whitespace inbetween words.
    Cull empty strings
    Cull stop words
    """
    input_str = preprocess_text(input_str)
    stop_words = load_stopwords()
    tokens = [s.lower() for s in input_str.split() if s != "" and s not in stop_words]
    tokens = stem_tokens(tokens)
    return tokens
