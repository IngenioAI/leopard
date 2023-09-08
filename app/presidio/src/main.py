import os
import json
import argparse

from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig, OperatorResult

from PIL import Image
from presidio_image_redactor import ImageRedactorEngine

'''
    example of params
    {
        type: 'analyze', 'anonymize', 'deanonymize', 'image_redact'
        input: 'string', 'image file path',
        entities: param's for analyzer
        language: en or kr?
        operators: anonymizer info
    }
'''


def result_to_dict(results):
    dict_result = []
    for result in results:
        dict_result.append(result.to_dict())
    return dict_result


def analyze(args, params):
    """
        개인정보 분석 함수.
    """
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=params['input'], entities=params['entities'], language=params['language'])
    with open("/data/output/%s" % args.output, "wt", encoding="UTF-8") as fp:
        json.dump(result_to_dict(results), fp)


def anonymize(args, params, operators):
    """
        익명화 함수.
    """
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=params['input'], entities=params['entities'], language=params['language'])

    engine = AnonymizerEngine()

    result = engine.anonymize(
        text=params['input'],
        analyzer_results=results,
        operators=operators
    )
    with open("/data/output/%s" % args.output, "wt", encoding="UTF-8") as fp:
        json.dump({
            'text': result.text,
            'items': result_to_dict(result.items)
        }, fp)


def de_anonymize(args, params, operators):
    """
        역 익명화 함수.
    """
    engine = DeanonymizeEngine()

    result_entities = []
    if 'result' in params:
        for result in params['result']:
            result_entities.append(OperatorResult(**result))

    result = engine.deanonymize(
        text=params['input'],
        entities=result_entities,
        operators=operators
    )
    print(result)
    with open("/data/output/%s" % args.output, "wt", encoding="UTF-8") as fp:
        json.dump({
            'text': result.text
        }, fp)


def image_redact(params):
    image = Image.open("/data/input/%s" % params['input'])
    engine = ImageRedactorEngine()
    redacted_image = engine.redact(image, (5, 5, 5))
    redacted_image.save("/data/output/result.png")


def main(args):
    with open("/data/input/%s" % args.input, "rt", encoding="UTF-8") as fp:
        params = json.load(fp)

    f_type = params['type']
    operators = None
    if 'operators' in params:
        operators = dict()
        for key, config in params['operators'].items():
            operators[key] = OperatorConfig(config['type'], config['params'])

    if f_type == 'analyze':
        analyze(args, params)

    elif f_type == 'anonymize':
        anonymize(args, params, operators)

    elif f_type == 'deanonymize':
        de_anonymize(args, params, operators)

    elif f_type == 'image_redact':
        image_redact(params)
    else:
        return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
