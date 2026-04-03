"""
inspect_rag.py — Show TM and TB (glossary) retrieval results, 100% match
context-check status, and the LLM prompt for each source sentence.

Run with:
    conda run -n t_ragx python inspect_rag.py > rag_inspection.txt
or simply:
    python inspect_rag.py
"""

import os
import pandas as pd
import t_ragx
from t_ragx.models.OpenAIModel import OpenAIModel
from t_ragx.models.BaseModel import glossary_to_text, trans_mem_to_text, pretext_to_text, tm_context_to_text
from t_ragx.models.constants import LANG_BY_LANG_CODE

MEMORY_CSV_PATH = os.path.join(os.path.dirname(__file__), "zh_en_memory.csv")
GLOSSARY_CSV_PATH = os.path.join(os.path.dirname(__file__), "zh_en_glossary.csv")

EXAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we live and work.",
    "She walked along the riverbank as the sun set behind the mountains.",
    "How can we lower the CAC using First-Party Data?",
    "The DSP is integrated with the top three Ad Exchange platforms.",
    "We observed a drop in Impressions but an increase in CTR.",
    "The MCN will manage the KOL cooperation for the summer launch.",
    "Is Header Bidding supported by your current SSP?",
    "A strong SEO strategy can complement your Programmatic Advertising.",
    "The brand is focusing on Private Traffic to increase ROAS.",
    "Does the Ad Server support Dynamic Creative Optimization?",
    "Our Yield Management team recommended using a PMP.",
    "The ATD is responsible for the overall media strategy.",
    "We need to audit our DMP for better data accuracy.",
    "D2C brands usually have a lower Customer Acquisition Cost.",
    "Native Advertising should look like natural content.",
    "What is the average CPM for this specific Ad Inventory?",
    "The OpenRTB protocol helps maintain industry standards.",
    "Low Viewability is affecting our brand safety scores.",
    "The KOC community is growing faster than the KOL market.",
    "Auction Dynamics vary depending on the time of day.",
    "We are looking for a new Ad Network to expand our reach.",
    "Attribution Modeling is necessary for omnichannel marketing.",
    "The KOC community is growing faster than the KOL market.",
    "Auction Dynamics vary depending on the time of day.",
    "We are looking for a new Ad Network to expand our reach.",
]

SOURCE_LANG = 'en'
TARGET_LANG = 'zh'


def load_exact_match_memory(csv_path, source_lang, target_lang):
    """Load exact match memory with context (mirrors TRagx.load_exact_match_memory)."""
    df = pd.read_csv(csv_path)
    src_col = df[source_lang].tolist()
    tgt_col = df[target_lang].tolist()
    exact_match_memory = {}
    for i, (src, tgt) in enumerate(zip(src_col, tgt_col)):
        prev = [src_col[j] for j in range(max(0, i - 1), i)]
        nxt = [src_col[j] for j in range(i + 1, min(len(src_col), i + 2))]
        entry = {"target": tgt, "prev": prev, "next": nxt}
        exact_match_memory.setdefault(src, []).append(entry)
    return exact_match_memory



def main():
    # --- Model (prompt-building only, no API calls made) ---
    model = OpenAIModel(
        host="api.groq.com", port=443, endpoint="/openai/v1",
        protocol="https", model="llama-3.3-70b-versatile", api_key="dummy",
    )

    # --- Set up input processor (mirrors test.py) ---
    input_processor = t_ragx.processors.RapidFuzzInputProcessor()
    input_processor.load_local_glossary(GLOSSARY_CSV_PATH, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)
    input_processor.load_general_translation(MEMORY_CSV_PATH, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)

    # --- Load exact match memory with context ---
    exact_match_memory = load_exact_match_memory(MEMORY_CSV_PATH, SOURCE_LANG, TARGET_LANG)

    # --- Batch-retrieve TM hits for all sentences ---
    tm_results = input_processor.search_memory(
        EXAMPLE_SENTENCES,
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
    )

    # --- Per-sentence glossary hits ---
    tb_results = input_processor.batch_search_glossary(
        EXAMPLE_SENTENCES,
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
    )

    # --- Build repeat map (mirrors batch_translate logic) ---
    # First, identify which indices are NOT TM context matches
    non_tm_indices = []
    for idx, src in enumerate(EXAMPLE_SENTENCES):
        if src in exact_match_memory:
            # Long segments (>5 words) used directly — excluded from repeat detection
            if len(src.split()) > 5:
                continue
            actual_prev = EXAMPLE_SENTENCES[max(0, idx - 1):idx]
            actual_next = EXAMPLE_SENTENCES[idx + 1:idx + 2]
            if any(occ["prev"] == actual_prev and occ["next"] == actual_next
                   for occ in exact_match_memory[src]):
                continue  # TM context match — excluded from repeat detection
        non_tm_indices.append(idx)

    repeat_map = {}  # text → list of (idx, prev_1, next_1)
    for i in non_tm_indices:
        text = EXAMPLE_SENTENCES[i]
        prev_1 = EXAMPLE_SENTENCES[max(0, i - 1):i]
        next_1 = EXAMPLE_SENTENCES[i + 1:i + 2]
        repeat_map.setdefault(text, []).append((i, prev_1, next_1))

    # For each later occurrence, determine status
    repeat_status = {}  # idx → ("same_context"|"context_differs"|"long_segment", first_occ_idx)
    for text, occurrences in repeat_map.items():
        if len(occurrences) < 2:
            continue
        first_occ_idx, first_prev, first_next = occurrences[0]
        for orig_idx, prev_1, next_1 in occurrences[1:]:
            if len(text.split()) > 5:
                status = "long_segment"
            else:
                status = "same_context" if (prev_1 == first_prev and next_1 == first_next) else "context_differs"
            repeat_status[orig_idx] = (status, first_occ_idx)

    # --- Print report ---
    separator = "=" * 80
    for idx, (src, tm_hits, tb_hits) in enumerate(zip(EXAMPLE_SENTENCES, tm_results, tb_results)):
        print(separator)
        print(f"[{idx + 1:02d}] SRC: {src}")
        print()

        # --- 100% Match Context Check ---
        tm_context = None  # will be set if 100% match with context difference
        if src in exact_match_memory:
            # Long segments (>5 words) are specific enough — use TM directly without context check
            if len(src.split()) > 5:
                print(f"  [100% MATCH] Status: LONG SEGMENT (>5 words) — used directly")
                print(f"       TM translation: {exact_match_memory[src][0]['target']}")
                print(f"       -> Bypassing context check (no LLM call)")
                print()
                continue

            # Build actual context (1 before + 1 after)
            actual_prev = EXAMPLE_SENTENCES[max(0, idx - 1):idx]
            actual_next = EXAMPLE_SENTENCES[idx + 1:idx + 2]

            occurrences = exact_match_memory[src]
            context_matched = False
            matched_occ_idx = None
            for occ_idx, occ in enumerate(occurrences):
                if occ["prev"] == actual_prev and occ["next"] == actual_next:
                    context_matched = True
                    matched_occ_idx = occ_idx
                    break

            if context_matched:
                occ = occurrences[matched_occ_idx]
                print(f"  [100% MATCH] Status: CONTEXT MATCH")
                print(f"       TM translation: {occ['target']}")
                print(f"       Context matched occurrence #{matched_occ_idx + 1}:")
                print(f"         TM prev: {occ['prev']}")
                print(f"         TM next: {occ['next']}")
                print(f"         Actual prev: {actual_prev}")
                print(f"         Actual next: {actual_next}")
                print(f"       -> Using TM translation directly (no LLM call)")
                print()
                # Skip TM/TB/PROMPT sections for context matches
                continue
            else:
                proposal = occurrences[0]["target"]
                # 2 before + 2 after for LLM context
                prev_segments = EXAMPLE_SENTENCES[max(0, idx - 2):idx]
                next_segments = EXAMPLE_SENTENCES[idx + 1:idx + 3]
                tm_context = {
                    "proposal": proposal,
                    "prev_segments": prev_segments,
                    "next_segments": next_segments,
                }
                print(f"  [100% MATCH] Status: CONTEXT DIFFERS — LLM review needed")
                print(f"       TM proposal: {proposal}")
                for occ_idx, occ in enumerate(occurrences):
                    print(f"       TM context (occurrence #{occ_idx + 1}):")
                    print(f"         TM prev: {occ['prev']}")
                    print(f"         TM next: {occ['next']}")
                print(f"       Actual context:")
                print(f"         prev (1): {actual_prev}")
                print(f"         next (1): {actual_next}")
                print(f"         prev (2, for LLM): {prev_segments}")
                print(f"         next (2, for LLM): {next_segments}")
                print(f"       -> Sending to LLM with proposal + surrounding context")
                print()
        else:
            print("  [100% MATCH] Status: NO MATCH")
            print()

        # --- Repeat Segment Check ---
        if idx in repeat_status:
            status, first_occ_idx = repeat_status[idx]
            actual_prev_1 = EXAMPLE_SENTENCES[max(0, idx - 1):idx]
            actual_next_1 = EXAMPLE_SENTENCES[idx + 1:idx + 2]
            first_prev_1 = EXAMPLE_SENTENCES[max(0, first_occ_idx - 1):first_occ_idx]
            first_next_1 = EXAMPLE_SENTENCES[first_occ_idx + 1:first_occ_idx + 2]
            if status == "long_segment":
                print(f"  [REPEAT] Status: LONG SEGMENT (>5 words) — copy translation from occurrence at [{first_occ_idx + 1:02d}]")
                print(f"       -> Bypassing context check (no LLM call)")
                print()
                continue
            elif status == "same_context":
                print(f"  [REPEAT] Status: SAME CONTEXT — copy translation from occurrence at [{first_occ_idx + 1:02d}]")
                print(f"       First occurrence prev: {first_prev_1}")
                print(f"       First occurrence next: {first_next_1}")
                print(f"       -> Using same translation directly (no LLM call)")
                print()
                # Skip TM/TB/PROMPT for same-context repeats
                continue
            else:
                prev_2 = EXAMPLE_SENTENCES[max(0, idx - 2):idx]
                next_2 = EXAMPLE_SENTENCES[idx + 1:idx + 3]
                tm_context = {
                    "proposal": f"<translation of occurrence [{first_occ_idx + 1:02d}]>",
                    "prev_segments": prev_2,
                    "next_segments": next_2,
                    "source": "repeat",
                }
                print(f"  [REPEAT] Status: CONTEXT DIFFERS — LLM review needed")
                print(f"       First occurrence at [{first_occ_idx + 1:02d}]")
                print(f"       First occurrence context:")
                print(f"         prev (1): {first_prev_1}")
                print(f"         next (1): {first_next_1}")
                print(f"       Current context:")
                print(f"         prev (1): {actual_prev_1}")
                print(f"         next (1): {actual_next_1}")
                print(f"         prev (2, for LLM): {prev_2}")
                print(f"         next (2, for LLM): {next_2}")
                print(f"       -> Sending to LLM with first occurrence's translation as proposal")
                print()
        else:
            if idx in non_tm_indices:
                # Only show NO REPEAT for first-occurrence or unique segments
                first_in_repeat = any(
                    occurrences[0][0] == idx
                    for occurrences in repeat_map.values()
                    if len(occurrences) > 1
                )
                if first_in_repeat:
                    print(f"  [REPEAT] Status: FIRST OCCURRENCE — translation will be reused for later repeats")
                    print()
                else:
                    print(f"  [REPEAT] Status: UNIQUE")
                    print()

        # --- Translation Memory ---
        print("  [TM] Translation Memory hits:")
        if not tm_hits:
            print("       (none)")
        else:
            for rank, hit in enumerate(tm_hits, 1):
                sim_pct = max(0, (1 - hit.get('normed_distance', 1)) * 100)
                print(f"       #{rank}  sim={sim_pct:.1f}%  score={hit.get('score', 0):.4f}  "
                      f"dist={hit.get('distance', '?')}  normed_dist={hit.get('normed_distance', '?'):.4f}")
                print(f"           EN: {hit.get(SOURCE_LANG, '')}")
                print(f"           ZH: {hit.get(TARGET_LANG, '')}")

        print()

        # --- Terminology Base ---
        print("  [TB] Terminology Base (glossary) hits:")
        if not tb_hits:
            print("       (none)")
        else:
            for term, translations in tb_hits.items():
                # translations may be a numpy array or list
                trans_list = list(translations) if hasattr(translations, '__iter__') and not isinstance(translations, str) else [translations]
                print(f"       '{term}'  →  {trans_list}")

        # --- LLM Prompt ---
        prompt = model.build_prompt(
            src,
            source_lang_code=SOURCE_LANG,
            target_lang_code=TARGET_LANG,
            search_result={'memory': tm_hits, 'glossary': tb_hits},
            tm_context=tm_context,
        )
        print("  [PROMPT] LLM input:")
        for msg in prompt:
            print(f"    [{msg['role']}]: {msg['content']}")
        print()

    print(separator)
    print(f"\nTotal sentences: {len(EXAMPLE_SENTENCES)}")


if __name__ == "__main__":
    main()
