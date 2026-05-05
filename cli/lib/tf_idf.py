import os
from pickle import dump

from .keyword_search import tokenize_text
from .utils_search import load_movies, PATH_CACHE


class InvertedIndex:
    def __init__(self):
        # dict mapping tokens to sets of ints
        self.index: dict[str, set[int]] = {}
        # dict mapping doc IDs to their full document objects
        self.docmap: dict[int, str] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """
        Tokenize the input text, then add each token to the index with the document ID
        """
        text_tokenized = tokenize_text(text)
        for token in text_tokenized:
            if token not in self.index.keys():
                self.index[token] = {doc_id}
            else:
                self.index[token].add(doc_id)
        return

    def get_documents(self, term: str) -> list[int]:
        """
        Get the set of document IDs for a given token, and return them as a
        list, sorted in ascending order.
        Assuming that the input term is a single word or token.
        """
        documents = self.index[term.lower()]
        return sorted(list(documents))

    def build(self) -> None:
        """
        Iterate over all indexes with `__add_document()`, concatenate both
        the title and description for use as input text.
        """
        # TODO: Hard coded data sources and field for now. Need to create more flexible API
        data = load_movies()
        field_1 = "title"
        field_2 = "description"

        for id, datum in enumerate(data, 1):
            tk = f"{datum[field_1]} {datum[field_2]}"
            self.__add_document(id, tk)
            self.docmap[id] = datum
            print(f"Added: {id}. {datum[field_1]}")

        return

    def save_cache(self, path_dst: str = PATH_CACHE):
        """
        Save the index and docmap attributes to disk using pickle module's dump
        function. Destination path defaults to /cache/.
        """
        path_dst = os.path.normpath(os.path.abspath(path_dst))

        if not os.path.exists(path_dst):
            print(path_dst)
            os.mkdir(path_dst)

        try:
            path_to_index_pkl = os.path.join(path_dst, "index.pkl")
            path_to_docmap_pkl = os.path.join(path_dst, "index.pkl")
            with open(path_to_index_pkl, "wb") as file:
                dump(self.index, file)
                print(f"Save to disk at : {os.path.join(path_dst)}")

            with open(path_to_docmap_pkl, "wb") as file:
                dump(self.docmap, file)
                print(f"Save docmap.pkl to disk at : {path_dst}/")
        except Exception as e:
            print(f"Error while saving: {e}")
