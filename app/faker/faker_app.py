import os
import json

from app.base import App


class FakerApp(App):
    def __init__(self, name="Faker"):
        super().__init__(name)
        self.config = {
            'name': name,
            'image': {
                'tag': 'leopard/faker:latest',
                'build': {
                    'base': "python:3.8",
                    'update': True,
                    'apt': "",
                    'pip': "faker",
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
        self.config['execution']['src'] = os.path.abspath("app/faker/src")
        self.config['execution']['input'] = os.path.abspath("storage/0/app/faker/data")
        self.config['execution']['output'] = os.path.abspath("storage/0/app/faker/run")
        with open("storage/0/app/faker/data/params.json", "wt", encoding="UTF-8") as fp:
            json.dump(params, fp)
        super().run(wait=True)
        with open("storage/0/app/faker/run/result.csv", "rt", encoding="UTF-8") as fp:
            return fp.read()


if __name__ == "__main__":
    app = FakerApp()
    app.run({})
