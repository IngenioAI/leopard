import os
import json
import importlib
from app.base import App


class AppManager():
    def __init__(self) -> None:
        with open(os.path.join("app", "appinfo.json"), "rt", encoding="utf-8") as fp:
            self.apps = json.load(fp)

    def start(self):
        for app_info in self.apps:
            if 'module' in app_info and 'class' in app_info:
                m = importlib.import_module(app_info['module'])
                c = getattr(m, app_info['class'])
                o = c(app_info)
            else:
                o = App(app_info)
            app_info['object'] = o
            if app_info['type'] == 'server':
                o.run()

    def run(self, module_id, params):
        for app_info in self.apps:
            if app_info['id'] == module_id:
                if app_info['type'] == 'server':
                    return app_info['object'].call_server(params)
                else:
                    return app_info['object'].run(params)
        return None

    def stop(self):
        for app_info in self.apps:
            if app_info['type'] == 'server':
                app_info['object'].stop()
