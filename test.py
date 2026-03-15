import os
import t_ragx
from elasticsearch import Elasticsearch

# Get your free API key at https://console.groq.com (no credit card required)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Path to the local zh-en translation memory CSV
MEMORY_CSV_PATH = os.path.join(os.path.dirname(__file__), "zh_en_memory.csv")
MEMORY_INDEX = "zh_en_translation_memory"
LOCAL_ES_HOST = "http://localhost:9200"

EXAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we live and work.",
    "She walked along the riverbank as the sun set behind the mountains.",
]


def index_memory_csv(es_client):
    """Index zh_en_memory.csv into local Elasticsearch if not already indexed."""
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
    # --- Input processor: glossary + translation memory ---
    input_processor = t_ragx.processors.ElasticInputProcessor()

    # Load en→zh glossary (downloads and caches a Parquet from S3)
    input_processor.load_general_glossary(source_lang='en', target_lang='zh')

    # --- Set up local zh-en translation memory ---
    # Requires Elasticsearch running locally:
    #   docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.11.0
    es_client = Elasticsearch(LOCAL_ES_HOST)
    index_memory_csv(es_client)
    input_processor.load_general_translation(
        elastic_index=MEMORY_INDEX,
        elasticsearch_host=LOCAL_ES_HOST,
        es_client=es_client,
    )

    # --- Model: Groq (free, OpenAI-compatible API) ---
    # Other free Groq models: "llama-3.1-8b-instant", "mixtral-8x7b-32768"
    model = t_ragx.models.OpenAIModel(
        host="api.groq.com",
        port=443,
        endpoint="/openai/v1",
        protocol="https",
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
    )

    # --- Translator ---
    translator = t_ragx.TRagx([model], input_processor=input_processor)

    # Load exact match memory — 100% matches bypass the LLM entirely
    translator.load_exact_match_memory(MEMORY_CSV_PATH, source_lang='en', target_lang='zh')

    # --- Translate ---
    translations = translator.batch_translate(
        EXAMPLE_SENTENCES,
        source_lang_code='en',
        target_lang_code='zh',
        search_memory=True,  # uses local zh-en translation memory
    )

    print("\n=== English → Chinese translations ===\n")
    for src, tgt in zip(EXAMPLE_SENTENCES, translations):
        print(f"  EN: {src}")
        print(f"  ZH: {tgt}")
        print()


if __name__ == "__main__":
    main()
