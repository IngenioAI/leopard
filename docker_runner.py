import os
import docker
import io
import threading
from ast import literal_eval
import re
import time


def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)


class DockerRunner():
    def __init__(self):
        self.client = docker.APIClient()
        self.threads = dict()

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
        self.threads[name] = dict()

        thread = threading.Thread(target=lambda: self.generator_thread(name, gen))
        self.threads[name]['thread'] = thread
        thread.start()
        self.threads[name]['status'] = 'running'

    def create_image(self, name, base_image="python:3.8", update=True, apt_install=None, pip_install=None,
                     additional_cmd=None):
        dockerfile_template = "FROM %s\n" % base_image
        if update:
            # dockerfile_template += "RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && apt update && apt -y upgrade\n"
            dockerfile_template += "RUN apt update && apt -y upgrade\n"
        if apt_install is not None and apt_install != '':
            dockerfile_template += "RUN apt install -y --allow-downgrades %s\n" % apt_install
        if pip_install is not None and pip_install != '':
            dockerfile_template += "RUN pip install --upgrade pip && pip install %s\n" % pip_install
        if additional_cmd is not None and additional_cmd != '':
            if type(additional_cmd) is list:
                for cmd in additional_cmd:
                    dockerfile_template += "RUN %s\n" % cmd
            else:
                commands = additional_cmd.split('\n')
                for cmd in commands:
                    dockerfile_template += "RUN %s\n" % cmd

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
        else:
            return {
                'lines': [],
                'status': 'not found'
            }

    def remove_create_image_info(self, name):
        del self.threads[name]

    def remove_image(self, name):
        try:
            self.client.remove_image(name);
            return True, None
        except (docker.errors.APIError, docker.errors.NotFound) as e:
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }

    def list_execs(self):
        return self.client.containers(all=True)

    def exec_command(self, src_dir, command, image, data_dir=None, output_dir=None, options={}):
        working_dir = "/app"
        binds = []
        if src_dir != '':
            binds.append('%s:%s' % (os.path.abspath(src_dir), working_dir))
        if data_dir is not None and data_dir != '':
            binds.append('%s:%s' % (os.path.abspath(data_dir), "/data/input"))
        if output_dir is not None and output_dir != '':
            binds.append('%s:%s' % (os.path.abspath(output_dir), "/data/output"))
        if "run_path" in options:
            binds.append("%s:%s" % (os.path.abspath(options["run_path"]), "/apprun"))

        #print("Binds:", binds)
        if type(command) == str:
            command_list = command.split(" ")
        elif type(command) == list:
            command_list = command
        else:
            print("Unsupported command type (string or list):", command)

        try:
            container = self.client.create_container(image, command=command_list,
                                                    working_dir=working_dir,
                                                    ports=[options['port']] if 'port' in options else [],
                                                    host_config=self.client.create_host_config(
                                                        device_requests=[
                                                            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                                                        binds=binds,
                                                        port_bindings={options['port']: options['port']} if 'port' in options else {}
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
            return False, {
                "error_code": e.response.status_code,
                "error_reason":e.response.reason,
                "error_message":  e.explanation
            }
