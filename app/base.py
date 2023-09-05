from docker_runner import DockerRunner
import time

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
        ret = self.docker.create_image(self.config['image']['tag'], build_info['base'], build_info['update'], build_info['apt'], build_info['pip'],
                                       build_info['additional_command'] if 'additional_command' in build_info else None)
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
        print(exec_info)
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