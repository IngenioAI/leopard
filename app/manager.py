import os
import json
import importlib
from threading import Timer

from app.base import App


class AppManager():
    def __init__(self) -> None:
        self.app_path = "app"
        self.app_info = []
        self.apps = {}
        self.monitor_timer = None

    def start(self, config):
        if config is not None and "app_path" in config:
            self.app_path = config["app_path"]

        app_dirs = os.listdir(self.app_path)
        for app_dir in app_dirs:
            appinfo_dir = os.path.join(self.app_path, app_dir)
            appinfo_path = os.path.join(appinfo_dir, "appinfo.json")
            if os.path.isdir(appinfo_dir) and os.path.exists(appinfo_path):
                with open(appinfo_path, "rt", encoding="utf-8") as fp:
                    appinfo = json.load(fp)
                    if not isinstance(appinfo, list):
                        appinfo = [appinfo]
                    for info in appinfo:
                        if info.get("enable", True):
                            self.app_info.append(info)

        for info in self.app_info:
            if 'module' in info and 'class' in info:
                m = importlib.import_module(info['module'])
                c = getattr(m, info['class'])
                o = c(info)
            else:
                o = App(info)
            self.apps[info['id']] = o

        self.monitor_start()

    def run(self, module_id, params):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].run(params)
        return None

    def get_progress(self, module_id):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].get_progress()
        return None

    def get_logs(self, module_id):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].logs()
        return None

    def get_result(self, module_id):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].get_result()
        return None

    def get_data(self, module_id, data_path):
        for info in self.app_info:
            if info['id'] == module_id:
                if hasattr(self.apps[module_id], 'get_data'):
                    return self.apps[module_id].get_data(data_path)
                else:
                    print("No get_data in module", module_id)
                    return None, None
        return None, None

    def stop_app(self, module_id):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].stop(remove=False)
        return None

    def remove_app(self, module_id):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].remove()
        return None

    def stop(self):
        for info in self.app_info:
            if info['type'] == 'server':
                self.apps[info['id']].stop()
        self.monitor_stop()

    def monitor_start(self):
        if self.monitor_timer is None:
            self.monitor_timer = Timer(1.0, self.monitor_run)
            self.monitor_timer.start()

    def monitor_stop(self):
        self.monitor_timer.cancel()
        self.monitor_timer = None

    def monitor_run(self):
        for info in self.app_info:
            app = self.apps[info['id']]
            if not app.is_running():
                if app.container_id is not None:
                    print("container remains:", info['id'])
                    app.logs()  # fetch and cache logs
                    app.remove()

        # repeat timer if not canceled
        if self.monitor_timer is not None:
            self.monitor_timer = Timer(1.0, self.monitor_run)
            self.monitor_timer.start()


app_manager = AppManager()
