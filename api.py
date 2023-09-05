import docker_runner
import io
import time
from ast import literal_eval

tested_image_list = [
    "python:3.8",
    "nvidia/cuda:10.2-base",
    "nvidia/cuda:11.4.1-base-ubuntu20.0",
    "tensorflow/tensorflow:2.11.0-gpu",
    "tensorflow/tensorflow:1.15.5-gpu",
    "pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel",
]

def init(args):
    pass

def server_log(tag, msg):
    print(tag, msg)

def list_images():
    client = docker_runner.APIClient()
    images = client.images()
    return images

def build_image(name, base_image="python:3.8", update=True, apt_install=None, pip_install=None, additional_cmd=None):
    dockerfile_template = "FROM %s\n" % base_image
    if update:
        dockerfile_template += "RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && apt update && apt -y upgrade\n"
    if apt_install is not None:
        dockerfile_template += "RUN apt install -y --allow-downgrades %s\n" % apt_install
    if pip_install is not None:
        dockerfile_template += "RUN pip install --upgrade pip && pip install %s\n" % pip_install
    if additional_cmd is not None:
        if type(additional_cmd) is list:
            for cmd in additional_cmd:
                dockerfile_template += "RUN %s\n" % cmd
        else:
            dockerfile_template += "RUN %s\n" % additional_cmd

    dockerfile = io.BytesIO(dockerfile_template.encode('utf-8'))
    client = docker_runner.APIClient()
    for line in client.build(fileobj=dockerfile, tag=name, rm=True, forcerm=True):
        line_info = literal_eval(line.decode('utf-8'))
        if 'stream' in line_info:
            try:
                print(line_info['stream'], end='')
            except UnicodeEncodeError as e:
                print(e)
        else:
            print(line_info)

def exec_python(src_dir, main_src, image, data_dir=None, output_dir=None):
    client = docker_runner.APIClient()
    ps_start_time = time.time()
    working_dir = "/app"
    binds = []
    binds.append('%s:%s' % (src_dir, working_dir))
    if data_dir is not None:
        binds.append('%s:%s' % (data_dir, "%s/data" % working_dir))
    if output_dir is not None:
        binds.append('%s:%s' % (output_dir, "%s/output" % working_dir))

    container = client.create_container(image, command="python %s" % main_src,
                                working_dir=working_dir,
                                host_config=client.create_host_config(
                                    auto_remove=True,
                                    device_requests=[docker_runner.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
                                    binds=binds
                                ))
    client.start(container.get('Id'))
    for log in client.logs(container.get('Id'), stream=True, since=ps_start_time):
        yield log.decode('utf-8')


if __name__ == "__main__":
    client = docker_runner.APIClient()

    setting = {
        'tag': "my-tf:0.1",
        'base': "tensorflow/tensorflow:2.11.0-gpu",
        'update': True,
        'apt': "libcudnn8=8.2.4.15-1+cuda11.4 libgl1-mesa-glx libglib2.0-0",
        'pip': "opencv-python mtcnn",
        'src': "/home/hkroh/work/leopard/mtcnn",
        'main': 'main.py'
    }

    image_found = False
    images = client.images()
    for im in images:
        # print(im['Id'][7:19], im['RepoTags'], "%.2fMB" % (im['Size']/1024/1024))
        if setting['tag'] in im['RepoTags']:
            image_found = True

    if not image_found:
        build_image(setting['tag'], setting['base'], update=setting['update'], apt_install=setting['apt'], pip_install=setting['pip'])

    logs = exec_python(setting['src'], setting['main'], setting['tag'])

    print("=== OUTPUT ===")
    for log in logs:
        print(log, end='')

'''
docker run --gpus all --name test -it my-tf /bin/bash
'''