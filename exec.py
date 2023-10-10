import docker_runner
import data_store

class ExecManager():
    def __init__(self) -> None:
        self.docker = docker_runner.DockerRunner()
        self.exec_list = []
        self.load()

    def load(self):
        self.exec_list = data_store.manager.get_data_list("exec")

    def save(self):
        data_store.manager.save_data_list("exec", self.exec_list)

    def get_list(self):
        current_exec_list = [self.get_info(x['id']) for x in self.exec_list]
        return current_exec_list

    def create_exec(self, id_name, source_path, command_line, base_image, input_path, output_path):
        res, info = self.docker.exec_command(source_path, command_line, base_image, input_path, output_path)
        if res:
            container_id = info["container_id"]
            exec_info = self.docker.exec_inspect(container_id)
            data = {
                "id": id_name,
                "base_image": base_image,
                "command_line": command_line,
                "source_path": source_path,
                "input_path": input_path,
                "output_path": output_path,
                "name": exec_info["Name"],
                "date": exec_info["Created"],
                "container_id": container_id
            }
            self.exec_list.append(data)
            self.save()
            return True, data
        return False, info

    def get_info(self, exec_id):
        exec_info = {}
        for info in self.exec_list:
            if info["id"] == exec_id:
                exec_info = dict.copy(info)
                break
        if "container_id" in exec_info:
            docker_exec_info = self.docker.exec_inspect(exec_info["container_id"])
            exec_info["container"] = docker_exec_info
        return exec_info

    def get_logs(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                return self.docker.exec_logs(info["container_id"])
        return ""

    def stop(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                return self.docker.exec_stop(info["container_id"])
        return False, { "error_message": "ID not found: %s" % exec_id}

    def remove_exec(self, exec_id):
        for info in self.exec_list:
            if info["id"] == exec_id:
                res, error_info = self.docker.exec_remove(info["container_id"])
                if res or error_info["error_code"] == 404:
                    self.exec_list.remove(info)
                    self.save()
                return res, error_info
        return False, { "error_message": "ID not found: %s" % exec_id}



manager = ExecManager()