"""
inspect_rag.py — Show TM and TB (glossary) retrieval results and the LLM prompt
for each source sentence.

Run with:
    conda run -n t_ragx python inspect_rag.py > rag_inspection.txt
or simply:
    python inspect_rag.py
"""

import os
import t_ragx
from elasticsearch import Elasticsearch
from t_ragx.models.OpenAIModel import OpenAIModel
from t_ragx.models.BaseModel import glossary_to_text, trans_mem_to_text, pretext_to_text
from t_ragx.models.constants import LANG_BY_LANG_CODE

MEMORY_CSV_PATH = os.path.join(os.path.dirname(__file__), "zh_en_memory.csv")
GLOSSARY_CSV_PATH = os.path.join(os.path.dirname(__file__), "zh_en_glossary.csv")
MEMORY_INDEX = "zh_en_translation_memory"
LOCAL_ES_HOST = "http://localhost:9200"

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
    "Attribution Modeling is necessary for omnichannel marketing."
]

SOURCE_LANG = 'en'
TARGET_LANG = 'zh'


def index_memory_csv(es_client):
    from t_ragx.utils.elastic import csv_to_elastic
    try:
        es_client.indices.create(index=MEMORY_INDEX)
        print(f"Indexing {MEMORY_CSV_PATH} into Elasticsearch...")
        csv_to_elastic(MEMORY_CSV_PATH, id_key='en', es_client=es_client, index=MEMORY_INDEX)
        print("Indexing complete.")
    except Exception as e:
        if 'resource_already_exists' in str(e).lower():
            print(f"Index '{MEMORY_INDEX}' already exists, skipping indexing.")
        else:
            raise


def main():
    # --- Model (prompt-building only, no API calls made) ---
    model = OpenAIModel(
        host="api.groq.com", port=443, endpoint="/openai/v1",
        protocol="https", model="llama-3.3-70b-versatile", api_key="dummy",
    )

    # --- Set up input processor (mirrors test.py) ---
    input_processor = t_ragx.processors.ElasticInputProcessor()
    input_processor.load_local_glossary(GLOSSARY_CSV_PATH, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)

    es_client = Elasticsearch(LOCAL_ES_HOST)
    index_memory_csv(es_client)
    input_processor.load_general_translation(
        elastic_index=MEMORY_INDEX,
        elasticsearch_host=LOCAL_ES_HOST,
        es_client=es_client,
    )

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

    # --- Print report ---
    separator = "=" * 80
    for idx, (src, tm_hits, tb_hits) in enumerate(zip(EXAMPLE_SENTENCES, tm_results, tb_results), 1):
        print(separator)
        print(f"[{idx:02d}] SRC: {src}")
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
        )
        print("  [PROMPT] LLM input:")
        for msg in prompt:
            print(f"    [{msg['role']}]: {msg['content']}")
        print()

    print(separator)
    print(f"\nTotal sentences: {len(EXAMPLE_SENTENCES)}")


if __name__ == "__main__":
    main()
