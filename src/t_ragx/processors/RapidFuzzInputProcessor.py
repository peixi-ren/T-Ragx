import logging
from typing import List, Union

import pandas as pd
import torch
from Levenshtein import distance
from rapidfuzz import process, fuzz

from .BaseInputProcessor import BaseInputProcessor
from ..utils.heuristic import clean_text

logger = logging.getLogger("t_ragx")


def rerank_rapidfuzz_result(candidates, source_lang, search_term, top_k=5):
    """
    Re-rank rapidfuzz recall candidates by Levenshtein distance (ascending).
    candidates: list of dicts with source_lang and target_lang fields
    """
    if not candidates:
        return []

    for c in candidates:
        c['distance'] = distance(c[source_lang], search_term)

    candidates = sorted(candidates, key=lambda x: x['distance'])
    return candidates[:top_k]


def search_single_rapidfuzz(corpus, corpus_texts, search_term, source_lang, target_lang, top_k=10, max_item_len=500):
    """
    Use rapidfuzz WRatio to recall top_k candidates from the in-memory corpus.

    corpus: list of dicts [{source_lang: ..., target_lang: ...}, ...]
    corpus_texts: list of source strings (parallel to corpus, pre-extracted for speed)
    """
    if not corpus:
        return []

    hits = process.extract(search_term, corpus_texts, scorer=fuzz.WRatio, limit=top_k)

    candidates = []
    for matched_text, score, idx in hits:
        row = corpus[idx]
        if target_lang not in row:
            continue
        candidates.append({
            'score': score,
            source_lang: row[source_lang][:max_item_len],
            target_lang: row[target_lang][:max_item_len],
        })

    return candidates


class RapidFuzzInputProcessor(BaseInputProcessor):
    """
    Docker-free input processor that uses rapidfuzz for fuzzy TM recall
    and Levenshtein for precision re-ranking.

    Drop-in replacement for ElasticInputProcessor — same search_memory() interface.
    """

    def __init__(self, device=None):
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__()

        self._tm_corpus: list = []         # list of dicts
        self._tm_corpus_texts: list = []   # source strings, parallel to _tm_corpus
        self._tm_source_lang: str = ''
        self._tm_target_lang: str = ''

    def load_general_translation(self, csv_path: str, source_lang: str, target_lang: str, **kwargs):
        """
        Load a CSV translation memory file into RAM for rapidfuzz search.
        The CSV must have language codes as column headers (e.g. 'en', 'zh').

        Args:
            csv_path: Path to the CSV file
            source_lang: Source language column name (e.g. 'en')
            target_lang: Target language column name (e.g. 'zh')
        """
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[source_lang, target_lang])
        df[source_lang] = df[source_lang].apply(clean_text)
        df = df[df[source_lang].str.len() > 0]
        df = df.drop_duplicates(subset=[source_lang])
        df = df.reset_index(drop=True)

        self._tm_source_lang = source_lang
        self._tm_target_lang = target_lang
        self._tm_corpus = df[[source_lang, target_lang]].to_dict(orient='records')
        self._tm_corpus_texts = [row[source_lang] for row in self._tm_corpus]

        logger.info(f"RapidFuzzInputProcessor: loaded {len(self._tm_corpus)} TM entries from {csv_path}")

    def search_general_memory(self, *args, **kwargs):
        return self.search_memory(*args, **kwargs)

    def search_memory(self, text_list: Union[List[str], str], source_lang: str = None, target_lang: str = None,
                      top_k: int = 10, rerank_top_k: int = None, max_item_len: int = 500,
                      pbar: bool = False, **kwargs):
        """
        Search the in-memory TM using rapidfuzz recall + Levenshtein re-ranking.
        Returns the same format as ElasticInputProcessor.search_memory().
        """
        if isinstance(text_list, str):
            text_list = [text_list]

        if source_lang is None:
            source_lang = self._tm_source_lang
        if target_lang is None:
            target_lang = self._tm_target_lang
        if rerank_top_k is None:
            rerank_top_k = top_k

        text_list = [clean_text(t) for t in text_list]

        processed_output = []
        for query in text_list:
            candidates = search_single_rapidfuzz(
                self._tm_corpus, self._tm_corpus_texts, query,
                source_lang, target_lang, top_k=top_k, max_item_len=max_item_len
            )
            reranked = rerank_rapidfuzz_result(candidates, source_lang, query, top_k=rerank_top_k)

            query_len = max(len(query), 1)
            for r in reranked:
                r['normed_distance'] = r['distance'] / query_len

            processed_output.append(reranked)

        return processed_output
