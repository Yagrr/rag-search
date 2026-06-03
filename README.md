# RAG-search

## Description

A CLI search engine for movies - Retrieval-augmented generation (RAG)-powered search.

This project is an implementation of keyword, vector, semantic and LLM-enhanced
search, with a full RAG pipeline for searching through a toy dataset from
[Boot.dev](https://www.boot.dev)'s ["Learn Retrieval Augmented Generation"](https://www.boot.dev/courses/learn-retrieval-augmented-generation) course taught by [Isaac Flath](https://isaacflath.com/). This tool is built by following the aforementioned course.

This project is largely for my own personal use and learning, use at your own risk.

## Installation

1) Requirements:
    - Python version >=3.13
    - uv version >=0.11

2) Clone the repository locally and install dependencies:
    
    ```python
    git clone <URL>
    cd rag-search
    uv sync
    ```

3. Activate virtual environment

- Linux/Mac: `source .venv/bin/activate`

- Windows: `.venv/Scripts/activate`

For further details on how to use `uv`, read the documentation:  ["Using a virtual environment"](https://docs.astral.sh/uv/pip/environments/).


## File structure

By default, this project uses the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model via the `sentence_transformers` package. For LLM calls, the `gemma-4-31b-it` open model is used via the Google API (`genai` Python package). The [Gemma](https://ai.google.dev/gemma/docs/core) model is used as it can be used with a free-tier with less restrictions than Gemini, although this is subject to change. For cross-encoder re-ranking, the [ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) model is used. For multimodal search to convert images to text, the [clip-ViT-B-32](https://huggingface.co/docs/transformers/model_doc/clip) model is used to embed an input image.

This project is structured such that all code related to CLI commands are
run from the `./cli/` folder, and all internals are stored in `./cli/lib/`.
This tool searches from a dataset of movie titles and their description located in
`./data/movies.json`. 

Using anything else except keyword search requires sentence embeddings to be
built from `./data/movies.json` on first-usage; this requires some time before search operations can be
performed. Text embeddings built from the `./data/movies.json` dataset are saved in
the `./cache/` folder, which is automatically created if it does not exist yet.
Subsequent searches are faster once embeddings are created.

## Basic Usage


> **IMPORTANT - Read if you choose to use this tool**: A Google API key is required for any LLM calls. If you
> would like to use all features from this project, please create a free API key via
> the [Google AI Studio](https://aistudio.google.com/). This project takes no
> responsibility for any token usage costs that you may incur if you choose to
> use the Hugging Face or Google Gemini's paid API. You're on your own.

Commands for searching are presented herein. Further functionalities, search
settings, and search modifiers are provided but not listed. Read the source code in the `_cli.py`
files for further details. Default search settings are located in `./cli/lib/utils_search`.

The `--limit` flag can be used to limit the number of search results.

### Keyword search

- Keyword search: `uv run cli/keyword_search_cli.py search "" --limit 10`

- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) search: `uv run cli/keyword_search_cli.py bm25search "Alien" --limit 10`

### Semantic search

Cosine similarities are used for semantic search using the `sentence_transformers` package.
Basic semantic search is performed without chunking, which is susceptible to
semantic dilution and imprecise results.

- Basic semantic search without chunking: `uv run cli/semantic_search_cli.py search "movie about monster in space" --limit 10`

- Chunked semantic search: `uv run cli/semantic_search_cli.py search_chunked "movie about monster in space"  --limit 10`

### Hybrid search

Min-Max normalisation is used to combine both BM25 search with chunked semantic
search for weighted search. The [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) search uses the document BM25 and semantic rankings instead of their scores.

- Weighted search (50:50 BM25 and semantic search): `uv run cli/hybrid_search_cli.py weighted-search "monster in space"  --limit 10`

- Reciprocal Rank Fusion: `uv run cli/hybrid_search_cli.py rrf-search "monster in space"  --limit 10`

### Logging hybrid search results

Logging capabilities are only available for RRF-search. Use the `--debug` flag to activate logs.
Search results following RRF-search are saved in `./data/search.log`.

- Logging: `uv run cli/hybrid_search_cli.py rrf-search "monster in space"  --limit 10 --debug`

### LLM-powered query enhancements

Enhancements can be made to the query using an LLM before it is passed to the search engine.

- Spell correction: `uv run cli/hybrid_search_cli.py rrf-search "monsttrerrr in space" --limit 10 --enhance spell `


- Query rewriting: `uv run cli/hybrid_search_cli.py rrf-search "the movie where there is a big monster in space"  --limit 10 --enhance rewrite`


- Query expansion: `uv run cli/hybrid_search_cli.py rrf-search "monster space"  --limit 10 --enhance expand`



### LLM re-ranking

Re-ranking allows for more accurate search beyond keyword/simlarity search.

- Re-rank documents individually in the context of the query: `uv run cli/hybrid_search_cli.py rrf-search "movie about a monster in space" --rerank-method individual --limit 3`

- Re-rank multiple documents in one batch in the context of the query: `uv run cli/hybrid_search_cli.py rrf-search "movie about a monster in space" --rerank-method batch --limit 3`

- [Cross-encoder](https://sbert.net/examples/cross_encoder/applications/README.html) re-ranking: `uv run cli/hybrid_search_cli.py rrf-search "movie about a monster in space" --rerank-method cross_encoder --limit 3`


### Multimodal search - images and text

This tool provides multimodal search capabilities. To rewrite a query given an input image, the Google API is called and uses the `all-MiniLM-L6-v2` by default. For multimodal search,
the `clip-ViT-B-32` model is used to embed the image into vector space.
Embeddings for an input image must be built before search can be performed.

Images of format `.jpeg` must be placed in the `./data/` folder for this to work.

Rewrite query based on an input image: `uv run cli/describe_image_cli.py --image data/alien.jpeg --query "monster in space"`

Multimodal search: `uv run cli/multimodal_search_cli.py image_search "data/alien.jpeg"`

### Performance evaluation

The `./data/golden_dataset.json` is a curated test dataset for checking search performance.
The `./cli/evaluation_cli.py` file is used to determine the Precision@K, Recall and F1 score

Call `uv run cli/evaluation_cli.py --limit 5` to see the general performance of the search engine.
