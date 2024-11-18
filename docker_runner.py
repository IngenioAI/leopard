import os
import io
import threading
import shutil
from ast import literal_eval
import re
import docker


def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


class DockerRunner():
    def __init__(self):
        self.client = docker.APIClient()
        self.threads = {}

    def list_images(self):
        return self.client.images()

    def generator_thread(self, name, gen):
        self.threads[name]['lines'] = ''
        self.threads[name]['status'] = 'running'
        for lines in gen:
            temp = lines.decode('utf-8')
            temp = temp.split("\n")
            for line in temp:
                if line == "":
                    continue
                line_info = literal_eval(line)
                if 'stream' in line_info:
                    s = self.threads[name]['lines'] + line_info['stream']
                    self.threads[name]['lines'] = escape_ansi(s)
        self.threads[name]['status'] = 'exited'

    def start_create_log(self, name, gen):
        self.threads[name] = {}

        thread = threading.Thread(target=lambda: self.generator_thread(name, gen))
        self.threads[name]['thread'] = thread
        thread.start()
        self.threads[name]['status'] = 'running'

    def create_image(self, name, base_image="python:3.8", update=True, upgrade=True, apt_install=None, pip_install=None,
                     additional_cmd=None):
        dockerfile_template = f'FROM {base_image}\n'
        if update:
            # dockerfile_template += "RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && apt update && apt -y upgrade\n"
            if upgrade:
                dockerfile_template += "RUN apt update && apt -y upgrade\n"
            else:
                dockerfile_template += "RUN apt update\n"
        if apt_install is not None and apt_install != '':
            dockerfile_template += f'RUN apt install -y --allow-downgrades {apt_install}\n'
        if pip_install is not None and pip_install != '':
            dockerfile_template += f'RUN pip install --upgrade pip && pip install {pip_install}\n'
        if additional_cmd is not None and additional_cmd != '':
            if isinstance(additional_cmd, list):
                for cmd in additional_cmd:
                    dockerfile_template += f'RUN {cmd}\n'
            else:
                commands = additional_cmd.split('\n')
                for cmd in commands:
                    dockerfile_template += f'RUN {cmd}\n'

        # print("=====DOCKERFILE\n", dockerfile_template, "\n========")
        dockerfile = io.BytesIO(dockerfile_template.encode('utf-8'))
        res = self.client.build(fileobj=dockerfile, tag=name, rm=True, forcerm=True)
        self.start_create_log(name, res)
        return True

    def get_create_image_info(self, name):
        if name in self.threads:
            return {
                'lines': self.threads[name]['lines'],
                'status': self.threads[name]['status']
            }
        return {
            'lines': [],
            'status': 'not found'
        }

    def remove_create_image_info(self, name):
        del self.threads[name]

    def remove_image(self, name):
        try:
            self.client.remove_image(name)
            return True, None
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }

    def list_execs(self):
        return self.client.containers(all=True)

    def exec_command(self, src_dir, command, image, data_dir=None, output_dir=None, options=None):
        working_dir = "/app"
        binds = []
        if src_dir != '':
            binds.append(f'{os.path.abspath(src_dir)}:{working_dir}')
        if data_dir is not None and data_dir != '':
            binds.append(f'{os.path.abspath(data_dir)}:/data/input')
        if output_dir is not None and output_dir != '':
            binds.append(f'{os.path.abspath(output_dir)}:/data/output')
        if options is not None and "run_path" in options:
            try:
                if os.path.exists(options["run_path"]):
                    shutil.rmtree(options["run_path"])
                os.makedirs(options["run_path"])
            except OSError:
                pass

            binds.append(f'{os.path.abspath(options["run_path"])}:/apprun')

        if options is not None and "binds" in options:
            binds_list = options["binds"]
            for bind_info in binds_list:
                target_path, src_path = bind_info
                if not os.path.exists(src_path):
                    os.makedirs(src_path)
                binds.append(f'{os.path.abspath(src_path)}:{target_path}')

        #print("Binds:", binds)
        if isinstance(command, str):
            command_list = command.split(" ")
        elif isinstance(command, list):
            command_list = command
        else:
            print("Unsupported command type (string or list):", command)
            return False, {
                "error_reason": f"Unsupported command type (string or list): {command}"
            }

        try:
            use_gpu = options["use_gpu"] if options is not None and "use_gpu" in options else True
            port_number = options["port"] if options is not None and "port" in options else None
            container = self.client.create_container(image, command=command_list,
                                                    working_dir=working_dir,
                                                    ports=[port_number] if port_number is not None else [],
                                                    host_config=self.client.create_host_config(
                                                        device_requests=[
                                                            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])] if use_gpu else [],
                                                        binds=binds,
                                                        shm_size="1G",
                                                        port_bindings={port_number: port_number} if port_number is not None else {}
                                                    ))
            self.client.start(container.get('Id'))
            return True, { "container_id": container.get('Id') }
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }

    def exec_logs(self, container_id):
        try:
            return self.client.logs(container_id).decode('utf-8')
            # return self.client.logs(container.get('Id'), stream=True, since=ps_start_time)
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            print("exec_logs:", e)
            return ""

    def exec_inspect(self, container_id):
        try:
            return self.client.inspect_container(container_id)
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            print("exec_inspect:", e)
            return {}

    def exec_stop(self, container_id):
        try:
            self.client.stop(container_id)
            return True, None
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }

    def exec_remove(self, container_id):
        try:
            self.client.remove_container(container_id)
            return True, None
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            print("docker remove error", e)
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }
