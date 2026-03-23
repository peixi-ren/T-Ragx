from typing import List, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from t_ragx.processors import ElasticInputProcessor, BaseInputProcessor
from t_ragx.models.AggregationModel import CometAggregationModel
from t_ragx.models.BaseModel import BaseModel


class TRagx:
    """
    🦖🦖🦖
    Translation using LLM with Retrieval Augmented Generation (RAG)

    """
    aggregate_model = None

    def __init__(self,
                 generation_models: Union[BaseModel, List[BaseModel]],
                 aggregate_model=None,
                 input_processor: BaseInputProcessor = None,
                 ):
        """

        Args:
            generation_models: a list of the T-Ragx models
            aggregate_model: a model that would choose the best translation without references. Currently only support
                                t_ragx.models.AggregationModel.CometAggregationModel
            input_processor: a T-Ragx input processor
        """

        self.input_processor = input_processor
        if input_processor is None:
            self.input_processor = ElasticInputProcessor()

        self.generation_models = generation_models
        if not isinstance(generation_models, list):
            self.generation_models = [generation_models]

        if aggregate_model is not None:
            self.aggregate_model = aggregate_model
        else:
            if len(self.generation_models) > 1:
                self.aggregate_model = CometAggregationModel()

        self.exact_match_memory = {}

    def load_exact_match_memory(self, csv_path: str, source_lang: str, target_lang: str):
        """
        Load a CSV translation memory file for exact match lookup with context.
        Stores each occurrence with 1 preceding and 1 following source sentence
        for context matching. If context matches, the TM translation is used
        directly. If context differs, the segment is sent to the LLM for review.

        Args:
            csv_path: Path to the CSV file with language code column headers (e.g. 'en', 'zh')
            source_lang: Source language column name (e.g. 'en')
            target_lang: Target language column name (e.g. 'zh')
        """
        df = pd.read_csv(csv_path)
        src_col = df[source_lang].tolist()
        tgt_col = df[target_lang].tolist()
        self.exact_match_memory = {}
        for i, (src, tgt) in enumerate(zip(src_col, tgt_col)):
            prev = [src_col[j] for j in range(max(0, i - 1), i)]
            nxt = [src_col[j] for j in range(i + 1, min(len(src_col), i + 2))]
            entry = {"target": tgt, "prev": prev, "next": nxt}
            self.exact_match_memory.setdefault(src, []).append(entry)

    def __call__(self, *args, **kwargs):
        return self.translate(*args, **kwargs)

    def translate(self, text, pre_text: list = None,
                  search_glossary=True, search_memory=True,
                  memory_search_args: dict = None,
                  glossary_search_args: dict = None,
                  prompt_args: List[dict] = None,
                  generation_args: List[dict] = None):
        pass

    def _run_llm_pass(self,
                      pass_indices,
                      original_text_list,
                      original_pre_text_list,
                      original_tokenize_args,
                      tm_context_map,
                      source_lang_code,
                      target_lang_code,
                      search_memory,
                      search_glossary,
                      memory_search_args,
                      glossary_search_args,
                      batch_size,
                      prompt_args,
                      generation_args):
        """
        Run a single LLM translation pass over a subset of indices.
        Segments with an entry in tm_context_map skip fuzzy TM retrieval.
        Returns a dict {orig_index: translation}.
        """
        text_list = [original_text_list[i] for i in pass_indices]
        pre_text_list = [original_pre_text_list[i] for i in pass_indices]
        tokenize_args = [original_tokenize_args[i] for i in pass_indices]
        tm_context_list = [tm_context_map.get(i, None) for i in pass_indices]

        # Only run memory search for segments without a TM/repeat proposal
        memory_results = [[]] * len(text_list)
        if search_memory:
            no_proposal = [j for j, tc in enumerate(tm_context_list) if tc is None]
            if no_proposal:
                search_texts = [text_list[j] for j in no_proposal]
                hits = self.input_processor.search_memory(
                    search_texts, source_lang=source_lang_code, target_lang=target_lang_code,
                    **memory_search_args)
                for j, result in zip(no_proposal, hits):
                    memory_results[j] = result

        glossary_results = [[]] * len(text_list)
        if search_glossary:
            glossary_results = self.input_processor.batch_search_glossary(
                text_list, source_lang=source_lang_code, target_lang=target_lang_code,
                **glossary_search_args)

        generation_output_dict = {}
        for model_idx, generation_model, p_args, tok_args, gen_args in zip(
                range(len(self.generation_models)),
                self.generation_models,
                prompt_args,
                tokenize_args,
                generation_args
        ):
            translated_text_list = []
            for batch_idx in tqdm(
                    np.array_split(list(range(len(text_list))), int(max(len(text_list) / batch_size, 1)))
            ):
                batch_text = [text_list[i] for i in batch_idx]
                batch_pre_text = [pre_text_list[i] for i in batch_idx]
                batch_tm_context = [tm_context_list[i] for i in batch_idx]
                batch_search_result = [
                    {'memory': memory_results[i], 'glossary': glossary_results[i]}
                    for i in batch_idx
                ]
                translated_text_list += generation_model.batch_translate(
                    batch_text,
                    source_lang_code=source_lang_code,
                    target_lang_code=target_lang_code,
                    batch_search_result=batch_search_result,
                    batch_pre_text=batch_pre_text,
                    batch_tm_context=batch_tm_context,
                    tokenize_config=tok_args,
                    generation_config=gen_args
                )
            generation_output_dict[model_idx] = translated_text_list

        generation_output = generation_output_dict[0]
        if len(generation_output_dict) > 1:
            generation_output = self.aggregate_model.combine_preds(
                generation_output_dict, text_list, target_lang_code=target_lang_code
            )

        return {orig_i: translation for orig_i, translation in zip(pass_indices, generation_output)}

    def batch_translate(self,
                        text_list,
                        pre_text_list: list = None,
                        batch_size=1,
                        source_lang_code='ja',
                        target_lang_code='en',
                        search_glossary=True,
                        search_memory=True,
                        memory_search_args: dict = None,
                        glossary_search_args: dict = None,
                        tokenize_args: List[dict] = None,
                        prompt_args: List[dict] = None,
                        generation_args: List[dict] = None
                        ):

        if pre_text_list is None:
            pre_text_list = [None] * len(text_list)

        if memory_search_args is None:
            memory_search_args = {}
        if glossary_search_args is None:
            glossary_search_args = {}

        if prompt_args is None:
            prompt_args = [{}] * len(self.generation_models)

        if generation_args is None:
            generation_args = [{}] * len(self.generation_models)

        if tokenize_args is None:
            tokenize_args = [{}] * len(text_list)

        # --- Exact match with context checking ---
        original_text_list = text_list  # preserve for context lookups
        original_pre_text_list = pre_text_list
        original_tokenize_args = tokenize_args
        final_results = [None] * len(text_list)
        llm_indices = []
        tm_context_map = {}  # orig_index → tm_context dict (TM or repeat proposals)

        for i, text in enumerate(text_list):
            if text in self.exact_match_memory:
                actual_prev = text_list[max(0, i - 1):i]
                actual_next = text_list[i + 1:i + 2]

                context_matched = False
                for occ in self.exact_match_memory[text]:
                    if occ["prev"] == actual_prev and occ["next"] == actual_next:
                        final_results[i] = occ["target"]
                        context_matched = True
                        break

                if not context_matched:
                    llm_indices.append(i)
                    tm_context_map[i] = {
                        "proposal": self.exact_match_memory[text][0]["target"],
                        "prev_segments": original_text_list[max(0, i - 2):i],
                        "next_segments": original_text_list[i + 1:i + 3],
                        "source": "tm",
                    }
            else:
                llm_indices.append(i)

        if not llm_indices:
            return final_results

        # --- Repeat detection: find duplicate segments within llm_indices ---
        repeat_map = {}  # text → list of (orig_idx, prev_1, next_1)
        for i in llm_indices:
            text = original_text_list[i]
            prev_1 = original_text_list[max(0, i - 1):i]
            next_1 = original_text_list[i + 1:i + 2]
            repeat_map.setdefault(text, []).append((i, prev_1, next_1))

        first_pass_indices = []
        deferred_repeats = []  # (orig_idx, first_occ_idx, same_context)
        for text, occurrences in repeat_map.items():
            first_pass_indices.append(occurrences[0][0])
            for orig_idx, prev_1, next_1 in occurrences[1:]:
                first_occ_idx, first_prev, first_next = occurrences[0]
                same_context = (prev_1 == first_prev and next_1 == first_next)
                deferred_repeats.append((orig_idx, first_occ_idx, same_context))

        # --- First LLM pass: translate first occurrences and non-repeats ---
        first_pass_results = self._run_llm_pass(
            first_pass_indices, original_text_list, original_pre_text_list,
            original_tokenize_args, tm_context_map,
            source_lang_code, target_lang_code,
            search_memory, search_glossary,
            memory_search_args, glossary_search_args,
            batch_size, prompt_args, generation_args
        )
        for orig_idx, translation in first_pass_results.items():
            final_results[orig_idx] = translation

        # --- Fill same-context repeats directly from first occurrence ---
        second_pass_indices = []
        for orig_idx, first_occ_idx, same_context in deferred_repeats:
            if same_context:
                final_results[orig_idx] = first_pass_results[first_occ_idx]
            else:
                second_pass_indices.append(orig_idx)
                tm_context_map[orig_idx] = {
                    "proposal": first_pass_results[first_occ_idx],
                    "prev_segments": original_text_list[max(0, orig_idx - 2):orig_idx],
                    "next_segments": original_text_list[orig_idx + 1:orig_idx + 3],
                    "source": "repeat",
                }

        # --- Second LLM pass: review context-differ repeats ---
        if second_pass_indices:
            second_pass_results = self._run_llm_pass(
                second_pass_indices, original_text_list, original_pre_text_list,
                original_tokenize_args, tm_context_map,
                source_lang_code, target_lang_code,
                search_memory, search_glossary,
                memory_search_args, glossary_search_args,
                batch_size, prompt_args, generation_args
            )
            for orig_idx, translation in second_pass_results.items():
                final_results[orig_idx] = translation

        return final_results
