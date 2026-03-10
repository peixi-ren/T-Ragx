import os
import t_ragx

# Get your free API key at https://console.groq.com (no credit card required)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

EXAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we live and work.",
    "She walked along the riverbank as the sun set behind the mountains.",
]


def main():
    # --- Input processor: glossary + translation memory ---
    input_processor = t_ragx.processors.ElasticInputProcessor()

    # Load en→zh glossary (downloads and caches a Parquet from S3)
    input_processor.load_general_glossary(source_lang='en', target_lang='zh')

    # Load general translation memory from the demo Elasticsearch cluster
    input_processor.load_general_translation(
        elastic_index="general_translation_memory",
        elasticsearch_host=["https://t-ragx-fossil.rayliu.ca", "https://t-ragx-fossil2.rayliu.ca"]
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

    # --- Translate ---
    translations = translator.batch_translate(
        EXAMPLE_SENTENCES,
        source_lang_code='en',
        target_lang_code='zh',
        search_memory=False,  # demo ES cluster only has ja/en pairs, no zh
    )

    print("\n=== English → Chinese translations ===\n")
    for src, tgt in zip(EXAMPLE_SENTENCES, translations):
        print(f"  EN: {src}")
        print(f"  ZH: {tgt}")
        print()


if __name__ == "__main__":
    main()
