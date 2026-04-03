import os
import t_ragx

# Get your free API key at https://console.groq.com (no credit card required)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Path to the local zh-en translation memory and glossary CSV files
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
    "We are looking for a new Ad Network to expand our reach."
]


def main():
    # --- Input processor: glossary + translation memory ---
    input_processor = t_ragx.processors.RapidFuzzInputProcessor()

    # Load en→zh glossary from local CSV
    input_processor.load_local_glossary(GLOSSARY_CSV_PATH, source_lang='en', target_lang='zh')

    # Load en→zh translation memory from local CSV (no Docker required)
    input_processor.load_general_translation(MEMORY_CSV_PATH, source_lang='en', target_lang='zh')

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

    # Load exact match memory — context matches bypass the LLM;
    # 100% matches with different context go to LLM for review
    translator.load_exact_match_memory(MEMORY_CSV_PATH, source_lang='en', target_lang='zh')

    # Build preceding text context (2 sentences before each segment)
    pre_text_list = [
        EXAMPLE_SENTENCES[max(0, i - 2):i] or None
        for i in range(len(EXAMPLE_SENTENCES))
    ]

    # --- Translate ---
    translations = translator.batch_translate(
        EXAMPLE_SENTENCES,
        pre_text_list=pre_text_list,
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
