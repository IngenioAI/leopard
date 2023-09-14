import os
import shutil
import urllib
import json

from app.base import App
import storage_util


class MTCNNApp(App):
    def __init__(self, name="MTCNN"):
        super().__init__(name)
        self.config = {
            'name': name,
            'image': {
                'tag': 'leopard/mtcnn:latest',
                'build': {
                    'base': "tensorflow/tensorflow:2.11.0-gpu",
                    'update': True,
                    'apt': "libcudnn8=8.2.4.15-1+cuda11.4 libgl1-mesa-glx libglib2.0-0",
                    'pip': "opencv-python mtcnn fastapi uvicorn"
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

    def run(self, input_path):
        input_dir, input_filename = os.path.split(input_path)
        run_path = os.path.abspath("storage/0/app/mtcnn/run")
        self.config['execution']['src'] = os.path.abspath("app/mtcnn")
        self.config['execution']['input'] = os.path.abspath(input_dir)
        self.config['execution']['output'] = run_path
        self.config['execution']['command_params'] = ["--input", input_filename]
        super().run()
        with open(os.path.join(run_path, "result.json"), "rt", encoding="UTF-8") as fp:
            res = json.load(fp)
        return res

    def run_server(self):
        run_path = os.path.abspath("storage/0/app/mtcnn/run")
        self.config['execution']['src'] = os.path.abspath("app/mtcnn/src")
        self.config['execution']['main'] = 'server.py'
        self.config['execution']['input'] = run_path
        super().run(wait=False)

    def call_server(self, params):
        input_path = storage_util.get_storage_file_path(params['storageId'], params['storagePath'])
        input_dir, input_filename = os.path.split(input_path)
        run_path = os.path.abspath("storage/0/app/mtcnn/run")
        target_path = os.path.join(run_path, input_filename)
        if input_dir != run_path:
            shutil.copy(input_path, target_path)

        params['input_filename'] = input_filename
        with open("storage/0/app/mtcnn/run/params.json", "wt", encoding="UTF-8") as fp:
            json.dump(params, fp)

        with urllib.request.urlopen(
                'http://localhost:%s/api/run/%s' % (self.config['execution']['port'], 'params.json')) as fp:
            res = json.load(fp)
        return res
