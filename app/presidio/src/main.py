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

def main(args):
    with open("/data/input/%s" % args.input, "rt") as fp:
        params = json.load(fp)

    if 'operators' in params:
        operators = dict()
        for key, config in params['operators'].items():
            operators[key] = OperatorConfig(config['type'], config['params'])
    else:
        operators = None

    if params['type'] in ['analyze', 'anonymize']:
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=params['input'], entities=params['entities'], language=params['language'])
        if params['type'] == 'analyze':
            with open("/data/output/%s" % args.output, "wt") as fp:
                json.dump(result_to_dict(results), fp)
            return
        
        engine = AnonymizerEngine()

        result = engine.anonymize(
            text=params['input'],
            analyzer_results=results,
            operators=operators
        )
        with open("/data/output/%s" % args.output, "wt") as fp:
            json.dump({
                'text': result.text,
                'items': result_to_dict(result.items)
            }, fp)        
    elif params['type'] == 'deanonymize':
        engine = DeanonymizeEngine()

        result_entities =[]
        if 'result' in params:
            for result in  params['result']:
                result_entities.append(OperatorResult(**result))

        result = engine.deanonymize(
            text=params['input'],
            entities=result_entities,
            operators=operators
        )
        print(result)
        with open("/data/output/%s" % args.output, "wt") as fp:
            json.dump({
                'text': result.text
            }, fp)
    elif params['type'] == 'image_redact':
        image = Image.open("/data/input/%s" % params['input'])
        engine = ImageRedactorEngine()
        redacted_image = engine.redact(image, (5, 5, 5))
        redacted_image.save("/data/output/result.png")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")
    return parser.parse_args()            

if __name__ == "__main__":
    main(parse_arguments())