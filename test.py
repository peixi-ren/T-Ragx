import t_ragx


def main():
    input_processor = t_ragx.processors.ElasticInputProcessor()

    input_processor.load_general_glossary()
    input_processor.load_general_translation(elastic_index="general_translation_memory", elasticsearch_host=["https://t-ragx-fossil.rayliu.ca", "https://t-ragx-fossil2.rayliu.ca"])

    example_input = "私はティラノサウルスです。"
    glossary_results = input_processor.batch_search_glossary([example_input], max_k=5, source_lang='ja', target_lang='en')
    
    print(f"\n input: {example_input}")
    print(f"output: {glossary_results} \n")


if __name__ == "__main__":
    main()
