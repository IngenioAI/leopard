import os

from app import App

class FakerApp(App):
    def __init__(self, name="Faker"):
        super().__init__(name)
        self.config = {
            'name': name,
            'image': {
                'tag': 'faker:latest',
                'build': {
                    'base': "python:3.8",
                    'update': True,
                    'apt': "",
                    'pip': "faker fastapi uvicorn",
                }
            },
            'execution': {
                'src': "",
                'main': 'main.py',
                'command_params': [],
                'input': "",
                'output': "",
                'port': 12710
            }
        }

    def run(self):
        self.config['execution']['src'] = os.path.abspath("app/faker")
        super().run(wait=True)

if __name__ == "__main__":
    app = FakerApp()
    app.run()
