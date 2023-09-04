from docker_run import DockerRunner
import os
import time
import json
import shutil
import urllib.request

class App():
    def __init__(self, name):
        self.name = name
        self.docker = DockerRunner()
        self.config = {
            'name': self.name,
            'image': {
            },
            'execution': {

            }
        }
        self.exec_id = None

    def build_image(self, wait=True):
        build_info = self.config['image']['build']
        ret = self.docker.create_image(self.config['image']['tag'], build_info['base'], build_info['update'], build_info['apt'], build_info['pip'])
        if not ret:
            print("Image creation failed")
            return
        
        if wait:
            last_line = 0
            while ret:
                time.sleep(0.5)
                image_info = self.docker.get_create_image_info(self.config['image']['tag'])
                if len(image_info['lines']) > last_line:
                    print(image_info['lines'][last_line:])
                    last_line = len(image_info['lines'])
                ret = image_info['status'] == 'running'

        self.docker.remove_create_image_info(self.config['image']['tag'])

    def run(self, wait=True):
        print('Run app:', self.config['name'])
        images = self.docker.list_images()
        targetImage = None
        for image in images:
            if self.config['image']['tag'] in image['RepoTags']:
                targetImage = image
                break
        if targetImage is None:
            self.build_image(wait=True)

        # execute
        exec_info = self.config['execution']
        exec_id = self.docker.exec_python(exec_info['src'], exec_info['main'], self.config['image']['tag'], exec_info['input'], exec_info['output'], 
                                    exec_info['port'] if 'port' in exec_info else None,
                                    exec_info['command_params'] if 'command_params' in exec_info else None)
        
        if wait:
            status = True
            last_line = 0
            while status:
                time.sleep(0.5)
                info = self.docker.exec_inspect(exec_id)
                logs = self.docker.exec_logs(exec_id)
                if len(logs) > last_line:
                    print(logs[last_line:])
                    last_line = len(logs)
                status = info['State']['Running']
            self.docker.exec_remove(exec_id)
            print("Exec done")
        self.exec_id = exec_id
        return exec_id    

    def stop(self, remove=True):
        self.docker.exec_stop(self.exec_id)
        if remove:
            self.docker.exec_remove(self.exec_id)

    def logs(self):
        return self.docker.exec_logs(self.exec_id)
    
    def inspect(self):
        return self.docker.exec_inspect(self.exec_id)


class MTCNNApp(App):
    def __init__(self, name="MTCNN"):
        super().__init__(name)
        self.config = {
            'name': name,
            'image': {
                'tag': 'mtcnn:latest',
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
        with open(os.path.join(run_path, "result.json"), "rt") as fp:
            res = json.load(fp)
        return res

    def run_server(self):
        run_path = os.path.abspath("storage/0/app/mtcnn/run")
        self.config['execution']['src'] = os.path.abspath("app/mtcnn")
        self.config['execution']['main'] = 'server.py'
        self.config['execution']['input'] = run_path
        super().run(wait=False)

    def call_server(self, input_path):
        input_dir, input_filename = os.path.split(input_path)
        run_path = os.path.abspath("storage/0/app/mtcnn/run")
        target_path = os.path.join(run_path, input_filename)
        if input_dir != run_path:
            shutil.copy(input_path, target_path)
        with urllib.request.urlopen('http://localhost:%s/api/run/%s' % (self.config['execution']['port'], input_filename)) as fp:
            res = json.load(fp)
        return res

if __name__ == "__main__":
    mtcnn_app = MTCNNApp()
    mtcnn_app.run_server()
    while True:
        cmd = input("> ")
        if cmd == "exit":
            break
        elif cmd == "stop":
            mtcnn_app.stop()
        elif cmd == "logs":
            result = mtcnn_app.logs()
            print(result)
        elif cmd == "inspect":
            result = mtcnn_app.inspect()
            print(result)