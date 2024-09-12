import os
import json
import time
from urllib import request, error
from docker_runner import DockerRunner
import storage_util


class App():
    def __init__(self, config):
        self.config = config
        self.docker = DockerRunner()
        self.container_id = None
        self.server_online = False
        self.run_params = None
        self.last_logs = ""

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

    def check_image(self):
        images = self.docker.list_images()
        target_image = None
        for image in images:
            if self.config['image']['tag'] in image['RepoTags']:
                target_image = image
                break
        if target_image is None:
            self.build_image(wait=True)

    def run(self, params=None, wait=None):
        exec_info = self.config['execution']
        print(exec_info)

        self.last_logs = ""

        self.run_params = params

        if self.container_id is not None:
            self.remove()  # remove previous docker

        is_server = self.config['type'] == "server"
        if self.server_online:
            return self.call_server(params)

        progrss_info = self.get_progress()
        if progrss_info["status"] == "running":
            return {
                "success": False,
                "error_message": "App already running",
                "container_id": self.container_id
            }

        if wait is None:
            if 'wait' in exec_info:
                wait = exec_info['wait']
            else:
                wait = not is_server

        if 'main' not in exec_info:
            exec_info['main'] = 'server.py' if is_server else 'main.py'

        self.check_image()

        if params is not None and "input" in self.config["execution"]:
            storage_util.ensure_path(self.config['execution']['input'])
            with open(os.path.join(self.config['execution']['input'], "params.json"), "wt", encoding="UTF-8") as fp:
                json.dump(params, fp)

        if "command" not in exec_info:
            command_line = ["python", exec_info["main"]] + (
                exec_info["command_params"] if "command_params" in exec_info else [])
        else:
            command_line = exec_info["command"]

        options = {}
        for item in ['port', 'run_path', 'binds']:
            if item in exec_info:
                options[item] = exec_info[item]

        res, info = self.docker.exec_command(exec_info['src'], command_line, self.config['image']['tag'],
                                             exec_info.get('input', None), exec_info.get('output', None), options)
        if res:
            self.container_id = info["container_id"]
        else:
            print(res, info)
            self.container_id = None
            return ""

        if wait:
            status = True
            last_line = 0
            logs = ""
            while status:
                time.sleep(0.5)
                info = self.inspect()
                logs = self.logs()
                if len(logs) > last_line:
                    print(logs[last_line:])
                    last_line = len(logs)
                status = info['State']['Running']
            self.docker.exec_remove(self.container_id)

            return self.get_result()

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
            return {
                "success": False,
                "error_message": "Server cannot be started"
            }

        return {
            "success": True,
            "container_id": self.container_id
        }

    def call_server(self, params):
        req = request.Request(f"http://localhost:{self.config['execution']['port']}/api/run",
                              data=json.dumps(params).encode("UTF-8"))
        try:
            with request.urlopen(req) as resp:
                return json.load(resp)
        except error.HTTPError as e:
            print(e)
            logs = self.logs()
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
        req = request.Request(f"http://localhost:{self.config['execution']['port']}/api/ping")
        try:
            with request.urlopen(req) as resp:
                info = json.load(resp)
                return info["success"]
        except (error.HTTPError, error.URLError, ConnectionResetError) as e:
            print(e)
            return False

    def stop(self, remove=True):
        if self.container_id is not None:
            self.docker.exec_stop(self.container_id)
            if remove:
                self.remove()

    def remove(self):
        if self.container_id is not None:
            info = self.inspect()
            if info['State']['Running']:
                self.docker.exec_stop(self.container_id)
            self.docker.exec_remove(self.container_id)
            self.container_id = None

    def logs(self):
        if self.container_id is not None:
            logs = self.docker.exec_logs(self.container_id)
            if len(logs) > 0:
                self.last_logs = logs
        return self.last_logs

    def inspect(self):
        if self.container_id is not None:
            return self.docker.exec_inspect(self.container_id)
        return {}

    def is_running(self):
        info = self.inspect()
        if 'State' in info:
            return info['State']['Running']
        return False

    def get_progress(self):
        info = self.inspect()
        if "State" in info:
            status = info['State']['Running']
            if 'run_path' in self.config['execution']:
                progress_path = os.path.join(self.config['execution']['run_path'], "progress.json")
                if os.path.exists(progress_path):
                    with open(progress_path, "rt", encoding="utf-8") as fp:
                        try:
                            progress_info = json.load(fp)
                            if not status and progress_info["status"] == "running":
                                progress_info["status"] = "exited"
                        except (json.decoder.JSONDecodeError):
                            progress_info = {"status": "running", "message": "json error"}

                        return progress_info
            return {
                "status": "running" if status else "done"
            }

        return {
            "status": "none"
        }

    def get_result(self):
        output_data_path = os.path.join(self.config['execution']['output'], "result.json")
        if os.path.exists(output_data_path):
            with open(output_data_path, "rt", encoding="UTF-8") as fp:
                result = json.load(fp)
            if self.run_params is not None and self.run_params.get("with_log", False):
                result["log"] = self.logs()
            return result
        return {"success": False, "error_message": "output file not found", "log": self.logs()}
