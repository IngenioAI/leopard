import argparse
import json

from faker import Faker

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

def result_to_dict(results):
    dict_result = []
    for result in results:
        dict_result.append(result.to_dict())
    return dict_result

def save_result(obj, filename):
    with open(f"/data/output/{filename}", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)

def main(args):
    with open(f"/data/input/{args.input}", "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    module_name = input_params.get("module", "")
    if module_name == "faker":
        fake = Faker('ko_KR')
        generation_type = input_params.get("type", "address")
        generation_count = input_params.get("count", 10)
        if generation_type == "address":
            generated_data = [fake.address() for _ in range(generation_count)]
        elif generation_type == "name":
            generated_data = [fake.name() for _ in range(generation_count)]
        elif generation_type == "ssn":
            generated_data = [fake.ssn() for _ in range(generation_count)]
        elif generation_type == "phone_number":
            generated_data = [fake.phone_number() for _ in range(generation_count)]

        save_result({
                "input_param": input_params,
                "generated_data": generated_data
            }, args.output)
    elif module_name == "presidio":
        action_type = input_params.get("type", "analyze")
        text = input_params.get("text", "")
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"},
                    {"lang_code": "en", "model_name": "en_core_web_lg"},],
        }

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["ko", "en"]
        )
        results = analyzer.analyze(text=text, language="ko")

        if action_type == "anonymize":
            anonymizer = AnonymizerEngine()
            anonymized_text = anonymizer.anonymize(text, results)
        else:
            anonymized_text = None

        save_result({
                "input_param": input_params,
                "results": result_to_dict(results),
                "anonymized_text": anonymized_text.text if anonymized_text is not None else ""
            }, args.output)
    else:


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
