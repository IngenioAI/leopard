import os
import json
import importlib
from app.base import App


class AppManager():
    def __init__(self) -> None:
        self.app_path = "app"
        self.app_info = []
        self.apps = {}

    def start(self, config):
        if config is not None and "app_path" in config:
            self.app_path = config["app_path"]

        app_dirs = os.listdir(self.app_path)
        for app_dir in app_dirs:
            appinfo_dir = os.path.join(self.app_path, app_dir)
            appinfo_path = os.path.join(appinfo_dir, "appinfo.json")
            if os.path.isdir(appinfo_dir) and os.path.exists(appinfo_path):
                with open(appinfo_path, "rt", encoding="utf-8") as fp:
                    self.app_info.append(json.load(fp))

        for info in self.app_info:
            if 'module' in info and 'class' in info:
                m = importlib.import_module(info['module'])
                c = getattr(m, info['class'])
                o = c(info)
            else:
                o = App(info)
            self.apps[info['id']] = o

    def run(self, module_id, params):
        for info in self.app_info:
            if info['id'] == module_id:
                return self.apps[module_id].run(params)
        return None

    def stop(self):
        for info in self.app_info:
            if info['type'] == 'server':
                self.apps[info['id']].stop()

app_manager = AppManager()