import os
import argparse
import csv
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

def clear_result(filename):
    with open(f"/data/output/{filename}", "wt", encoding="UTF-8") as fp:
        json.dump({}, fp, indent=4)

def save_result(obj, filename):
    with open(f"/data/output/{filename}", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)

def generate_fake_data(generation_type, generation_count):
    fake = Faker('ko_KR')
    if generation_type == "address":
        generated_data = [fake.address() for _ in range(generation_count)]
    elif generation_type == "name":
        generated_data = [fake.name() for _ in range(generation_count)]
    elif generation_type == "ssn":
        generated_data = [fake.ssn() for _ in range(generation_count)]
    elif generation_type == "phone_number":
        generated_data = [fake.phone_number() for _ in range(generation_count)]
    elif generation_type == "credit_card_number":
        generated_data = [fake.credit_card_number(card_type=None) for _ in range(generation_count)]
    elif generation_type == "email":
        generated_data = [fake.email() for _ in range(generation_count)]
    elif generation_type == "passport_number":
        generated_data = [fake.passport_number() for _ in range(generation_count)]
    else:
        generated_data = []
    return generated_data

def run_faker(params, args):
    generation_type = params.get("generation_type", "address")
    generation_count = params.get("generation_count", 10)
    generated_data = generate_fake_data(generation_type, generation_count)
    save_result({
            "generated_data": generated_data
        }, args.output)

def run_presidio(params, args):
    action_type = params.get("action_type", "analyze")
    if action_type == "generate_and_analyze":
        base_dataset = params["generate_info"]["base"]
        field_list = params["generate_info"]["field"]
        count = params["generate_info"]["count"]
        generated_data_list = []
        for field in field_list:
            generated_data = generate_fake_data(field, count)
            if len(generated_data):
                generated_data_list.append(generated_data)
        csv_path = os.path.join("data", f"{base_dataset}.csv")
        output_path = os.path.join("/upload", f"{base_dataset}_output.csv")
        if not os.path.exists(csv_path):
            save_result({
                "error": "base data file not found"
            }, args.output)
            return
        c = 0
        with open(csv_path, newline='') as csvfile:
            with open(output_path, 'w', newline='') as outfile:
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                csv_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in csv_reader:
                    if c ==0 :
                        generated_header_list = []
                        for i in range(len(generated_data_list)):
                            generated_header_list.append(f"synth_feature_{i+1}")
                        csv_writer.writerow(row[:1] + generated_header_list + row[1:])
                    else:
                        data = []
                        for i in range(len(generated_data_list)):
                            data.append(generated_data_list[i][c-1])
                        csv_writer.writerow(row[:1] + data + row[1:])
                    c += 1
                    if c >= count:
                        break
        text_file = f"{base_dataset}_output.csv"
    elif action_type == "analyze":
        text = params.get("text", None)
        text_file = params.get("text_file", "")
    elif action_type == "anonymize":
        text = params.get("text", None)
        text_file = params.get("text_file", "")

    entities = params.get("entities", None)
    if text_file != "":
        text_file_path = os.path.join("/data/input", text_file)
        if not os.path.exists(text_file_path):
            text_file_path = os.path.join("/upload", text_file)
        with open(text_file_path, "rt", encoding="utf-8") as fp:
            text = fp.read()
            text = text.replace("\r", "")
        if len(text) >= 1000000:
            text = text[:1000000]
    if entities is None:
        entities = ["주민등록번호", "전화번호", "신용카드번호", "EMAIL_ADDRESS", "여권번호", "주소"]    # , "건강보험번호", "운송장번호", "운전면허번호"

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"},
                {"lang_code": "en", "model_name": "en_core_web_lg"},],
    }

    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["ko", "en"]
        )
        results = analyzer.analyze(text=text, entities=entities, language="ko")

        if action_type == "anonymize":
            anonymizer = AnonymizerEngine()
            anonymized_result = anonymizer.anonymize(text, results)
        else:
            anonymized_result = None

        save_result({
                "results": result_to_dict(results),
                "anonymized_text": anonymized_result.text if anonymized_result is not None else "",
                "text_content": text if action_type == "generate_and_analyze" else ""
            }, args.output)

    except ValueError as e:
        print(e)
        save_result({
            "input_param": params,
            "results": None,
            "anomymized_text": "",
            "error": str(e)
        }, args.output)


def main(args):
    with open(f"/data/input/{args.input}", "rt", encoding="utf-8") as fp:
        input_params = json.load(fp)

    module_name = input_params.get("module", "")
    if module_name == "faker":
        run_faker(input_params, args)
    elif module_name == "presidio":
        run_presidio(input_params, args)
    else:
        save_result({
            "module": "presidio",
            "type": "analyze",
            "text": "샘플 전화번호는 010-3212-6758입니다."
        }, args.output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
