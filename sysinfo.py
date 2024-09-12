import os
import sys
import platform
import subprocess
import xml.etree.ElementTree as XMLTree
import threading
import time

import psutil


class SystemInfo():
    def __init__(self):
        self.info = {}
        self.poll_time = 1.0
        self.thread = None
        self.stop_flag = False

    def start(self, config, poll_time=1.0):  # pylint: disable=unused-argument
        self.stop_flag = False
        self.poll_time = poll_time
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.stop_flag = True

    def run(self):
        while not self.stop_flag:
            system_info = {}
            system_info['cpu_info'] = platform.processor()
            system_info['cpu_thread_count'] = psutil.cpu_count()
            system_info['cpu_core_count'] = psutil.cpu_count(logical=False)
            system_info['cpu_util'] = psutil.cpu_percent()
            system_info['total_memory'] = psutil.virtual_memory()[0]
            system_info['available_memory'] = psutil.virtual_memory()[1]
            system_info['platform'] = platform.platform()
            system_info['node'] = platform.node()
            system_info['gpu_info'] = self.get_gpu_info()
            disk_info = psutil.disk_usage(".")
            system_info['disk_total'] = disk_info.total
            system_info['disk_used'] = disk_info.used
            system_info['disk_free'] = disk_info.free
            self.info = system_info
            time.sleep(self.poll_time)

    def get_system_info(self):
        return self.info

    def get_gpu_info(self):
        if sys.platform == 'win32':
            possible_path = [
                'C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe',
                'C:/Windows/system32/nvidia-smi.exe'
            ]
        else:
            possible_path = ['/usr/bin/nvidia-smi']

        nvidia_smi_path = ''
        for p in possible_path:
            if os.path.exists(p):
                nvidia_smi_path = p
                break

        if nvidia_smi_path == '':
            return []  # no gpu info

        nvidia_cmd = [nvidia_smi_path, '-x', '-q']

        try:
            with subprocess.Popen(nvidia_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as nvidia_process:
                stdout, _ = nvidia_process.communicate()
                tree_root = XMLTree.fromstring(stdout)
                gpu_info_list = []
                if tree_root.tag == 'nvidia_smi_log':
                    for gpu in tree_root.findall('gpu'):
                        gpu_info = {}
                        gpu_info['name'] = gpu.find('product_name').text
                        gpu_info['gpu_util'] = gpu.find('utilization').find('gpu_util').text
                        gpu_info['mem_util'] = gpu.find('utilization').find('memory_util').text
                        gpu_info['total_mem'] = gpu.find('fb_memory_usage').find('total').text
                        gpu_info['used_mem'] = gpu.find('fb_memory_usage').find('used').text
                        gpu_info['free_mem'] = gpu.find('fb_memory_usage').find('free').text
                        gpu_info['temp'] = gpu.find('temperature').find('gpu_temp').text
                        gpu_info['power'] = gpu.find('power_readings').find('power_draw').text
                        gpu_info['power_limit'] = gpu.find('power_readings').find('power_limit').text
                        gpu_info_list.append(gpu_info)
        except FileNotFoundError:
            gpu_info_list = []

        return gpu_info_list


sys_info = SystemInfo()
