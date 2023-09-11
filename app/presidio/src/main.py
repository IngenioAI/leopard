import os
import json
import argparse

from utils import *

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


@deprecated
def result_replace(params, results):
    # csv파일로 받고, 내려줄때도 csv파일로 내려서 parsing 작업 예정
    # 따라서 필요 없음.
    input_text = params['input']

    result_dict = result_to_dict(results)
    print(len(result_dict))
    # FIXME span 과 css 속성 값을 부여할 수도 있음.
    bold_start = '<b>'
    bold_end = '</b>'
    replaced_text = []
    for i in range(len(result_dict)):
        res = result_dict[i]

        start = res['start']
        end = res['end']
        result_text = input_text[0:start] + bold_start + input_text[start:end] + bold_end + input_text[end:] + '\n'
        replaced_text.append(result_text)

    return ''.join(replaced_text)


def analyze(args, params):
    """
        개인정보 분석 함수.
    """
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=params['input'], entities=params['entities'], language=params['language'])
    result_list = result_to_dict(results)

    # replaced_text = result_replace(params, results)

    # result = {'replaced_text': replaced_text, 'result': result_list}
    with open("/data/output/%s" % args.output, "wt", encoding="UTF-8") as fp:
        json.dump(result_list, fp)


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
