import os
import json
import importlib
from app.base import App


class AppManager():
    def __init__(self) -> None:
        with open(os.path.join("app", "appinfo.json"), "rt", encoding="utf-8") as fp:
            self.app_info = json.load(fp)
        self.apps = {}

    def start(self):
        for info in self.app_info:
            if 'module' in info and 'class' in info:
                m = importlib.import_module(info['module'])
                c = getattr(m, info['class'])
                o = c(info)
            else:
                o = App(info)
            self.apps[info['id']] = o
            if info['type'] == 'server':
                o.run()

    def run(self, module_id, params):
        for info in self.app_info:
            if info['id'] == module_id:
                if info['type'] == 'server':
                    return self.apps[module_id].call_server(params)
                else:
                    return self.apps[module_id].run(params)
        return None

    def stop(self):
        for info in self.app_info:
            if info['type'] == 'server':
                self.apps[info['id']].stop()
