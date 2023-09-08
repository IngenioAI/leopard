import os
import json

from app.base import App

class PresidioApp(App):
    def __init__(self, name="Presidio"):
        super().__init__(name)
        self.config = {
            'name': name,
            'image': {
                'tag': 'leopard/presidio:latest',
                'build': {
                    'base': "python:3.8",
                    'update': True,
                    'apt': "tesseract-ocr libtesseract-dev",
                    'pip': "presidio_analyzer presidio_anonymizer presidio_image_redactor Pillow",
                    'additional_command': "python -m spacy download en_core_web_lg"
                }
            },
            'execution': {
                'src': "",
                'main': 'main.py',
                'command_params': [],
                'input': "",
                'output': ""
            }
        }

    def run(self, params):
        self.config['execution']['src'] = os.path.abspath("app/presidio/src")
        self.config['execution']['input'] = os.path.abspath("storage/0/app/presidio/data")
        self.config['execution']['output'] = os.path.abspath("storage/0/app/presidio/run")
        with open("storage/0/app/presidio/data/params.json", "wt", encoding="UTF-8") as fp:
            json.dump(params, fp)
        super().run(wait=True)
        if params['type'] == 'image_redact':
            return { 'result_url': "0/app/presidio/run/result.png"}
        else:
            with open("storage/0/app/presidio/run/result.json", "rt", encoding="UTF-8") as fp:
                return json.load(fp)

if __name__ == "__main__":
    app = PresidioApp()
    app.run()
