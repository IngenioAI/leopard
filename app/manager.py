import importlib


class AppManager():
    def __init__(self) -> None:
        self.apps = [
            {
                'id': "mtcnn",
                'name': "MTCNN",
                'module': "app.mtcnn.mtcnn_app",
                'class': "MTCNNApp",
                'type': 'server'
            },
            {
                'id': "faker",
                'name': "Faker",
                'module': "app.faker.faker_app",
                'class': "FakerApp",
                'type': 'script'
            },
            {
                'id': "presidio",
                'name': "Presidio",
                'module': "app.presidio.presidio_app",
                'class': "PresidioApp",
                'type': 'script'
            }
        ]

    def run(self):
        for app_info in self.apps:
            m = importlib.import_module(app_info['module'])
            c = getattr(m, app_info['class'])
            o = c()
            app_info['object'] = o
            if app_info['type'] == 'server':
                o.run_server()

    def call(self, module_id, params):
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
