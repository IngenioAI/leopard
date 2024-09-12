from urllib import request
import time
from docker_runner import DockerRunner


class TensorBoard():
    def __init__(self) -> None:
        self.docker = DockerRunner()
        self.log_path = ""
        self.exec_info = None

    def start(self, log_path, port=12760, wait_timeout=10):
        if log_path != self.log_path:
            if self.exec_info is not None:
                self.stop()  # stop previous service

            success, exec_info = self.docker.exec_command(
                src_dir='',
                data_dir=log_path,
                command=f'tensorboard --logdir /data/input --port {port} --bind_all',
                image="tensorflow/tensorflow:2.13.0-gpu",
                options={"port": port, "use_gpu": True}
            )
            if success:
                self.exec_info = exec_info
                self.log_path = log_path

            tensorboard_url = f'http://127.0.0.1:{port}'

            if wait_timeout > 0:
                req = request.Request(tensorboard_url, method="HEAD")
                wait = True
                start_time = time.time()
                while wait:
                    try:
                        with request.urlopen(req) as _:
                            wait = False
                    except ConnectionError as e:
                        print(e)
                        if time.time() - start_time > wait_timeout:
                            return False
                        time.sleep(0.5)
        return True

    def logs(self):
        if self.exec_info is not None:
            return self.docker.exec_logs(self.exec_info["container_id"])
        return ""

    def stop(self):
        if self.exec_info is not None:
            success, _ = self.docker.exec_stop(self.exec_info["container_id"])
            if success:
                self.docker.exec_remove(self.exec_info["container_id"])
            self.exec_info = None


manager = TensorBoard()

if __name__ == "__main__":
    tb = TensorBoard()
    tb.start("storage/run/TEST/logs", 12799)
    cmd = ""  # pylint: disable=invalid-name
    while cmd != "exit":
        cmd = input("TB> ")
        if cmd == "logs":
            print(tb.logs())
    tb.stop()
