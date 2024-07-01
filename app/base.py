import os
import json
from urllib import request, error
from docker_runner import DockerRunner
import storage_util
import time


class App():
    def __init__(self, config):
        self.config = config
        self.docker = DockerRunner()
        self.container_id = None
        self.server_online = False

    def build_image(self, wait=True):
        build_info = self.config['image']['build']
        ret = self.docker.create_image(self.config['image']['tag'], build_info['base'], build_info['update'],
                                       build_info['apt'], build_info['pip'],
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

    def run(self, params=None, wait=None):
        print('Run app:', self.config['name'])
        exec_info = self.config['execution']
        print(exec_info)

        is_server = self.config['type'] == "server"
        if self.server_online:
            return self.call_server(params)

        if wait is None:
            if is_server:
                wait = False
            else:
                wait = True

        if 'main' not in exec_info:
            if is_server:
                exec_info['main'] = 'server.py'
            else:
                exec_info['main'] = 'main.py'

        images = self.docker.list_images()
        targetImage = None
        for image in images:
            if self.config['image']['tag'] in image['RepoTags']:
                targetImage = image
                break
        if targetImage is None:
            self.build_image(wait=True)

        if params is not None:
            storage_util.ensure_path(self.config['execution']['input'])
            with open(os.path.join(self.config['execution']['input'], "params.json"), "wt", encoding="UTF-8") as fp:
                json.dump(params, fp)

        if "command" not in exec_info:
            command_line = ["python", exec_info["main"]] + (exec_info["command_params"] if "command_params" in exec_info else [])
        else:
            command_line = exec_info["command"]
        res, info = self.docker.exec_command(exec_info['src'], command_line, self.config['image']['tag'],
                                        exec_info['input'], exec_info['output'],
                                        { "port": exec_info['port'] } if 'port' in exec_info else {})
        if res:
            self.container_id = info["container_id"]
        else:
            print(res, info)
            self.container_id = ""
            return ""

        if wait:
            status = True
            last_line = 0
            logs = ""
            while status:
                time.sleep(0.5)
                info = self.docker.exec_inspect(self.container_id)
                logs = self.docker.exec_logs(self.container_id)
                if len(logs) > last_line:
                    print(logs[last_line:])
                    last_line = len(logs)
                status = info['State']['Running']
            self.docker.exec_remove(self.container_id)

            output_data_path = os.path.join(self.config['execution']['output'], "result.json")
            if os.path.exists(output_data_path):
                with open(output_data_path, "rt", encoding="UTF-8") as fp:
                    result = json.load(fp)
                if params is not None and "with_log" in params and params["with_log"]:
                    result["log"] = logs
                return result
            else:
                return { "success": False, "error_message": "output file not found", "log": logs}

        if is_server:
            timeout = 30
            while timeout > 0:
                if self.ping_server():
                    break
                time.sleep(1.0)
                timeout -= 1
            print("Timeout:", timeout)
            if timeout > 0:
                self.server_online = True
                return self.call_server(params)
            else:
                return {
                    "success": False,
                    "error_message": "Server cannot be started"
                }

        return self.container_id

    def call_server(self, params):
        req = request.Request('http://localhost:%s/api/run' % (self.config['execution']['port']),
                              data=json.dumps(params).encode("UTF-8"))
        try:
            resp = request.urlopen(req)
            return json.load(resp)
        except error.HTTPError as e:
            print(e)
            logs = self.docker.exec_logs(self.container_id)
            return {
                "success": False,
                "error_message": e.reason,
                "log": logs
            }
        except error.URLError as e:
            print(e)
            return {
                "success": False,
                "error_message": e.reason
            }

    def ping_server(self):
        req = request.Request('http://localhost:%s/api/ping' % (self.config['execution']['port']))
        try:
            resp = request.urlopen(req)
            info = json.load(resp)
            return info["success"]
        except (error.HTTPError, error.URLError, ConnectionResetError) as e:
            print(e)
            return False

    def stop(self, remove=True):
        if self.container_id:
            self.docker.exec_stop(self.container_id)
            if remove:
                self.docker.exec_remove(self.container_id)

    def logs(self):
        return self.docker.exec_logs(self.container_id)

    def inspect(self):
        return self.docker.exec_inspect(self.container_id)
