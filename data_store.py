import os
import json

class DataStore():
    def __init__(self, name, type="list") -> None:
        self.name = name
        self.current_data = None
        self.type = type

    def load(self):
        dataset_info_path = os.path.join("data", "%s.json" % self.name)
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "rt", encoding="utf-8") as fp:
                self.current_data = json.load(fp)
        else:
            if self.type == "list":
                self.current_data = []
            else:
                self.current_data = {}

    def save(self):
        if not os.path.exists("data"):
            os.mkdir("data")
        dataset_info_path = os.path.join("data", "%s.json" % self.name)
        with open(dataset_info_path, "wt", encoding="utf-8") as fp:
            json.dump(self.current_data, fp, indent=4)
        return True

    def get(self):
        if self.current_data is None:
            self.load()
        return self.current_data

    def add(self, data):
        if self.type == "list":
            self.current_data.append(data)
            self.save()
        else:
            print("non-list type datastore add data with (key, value)")

    def set_value(self, key, value):
        if self.type == "single":
            if value is not None:
                self.current_data[key] = value
            else:
                del self.current_data[key]
            self.save()
        else:
            print("list type datastore cannot set_value")

    def get_value(self, key, default=None):
        if key in self.current_data:
            return self.current_data[key]
        return default

    def set_all(self, data):
        self.current_data = data
        self.save()

    def get_item(self, key, value):
        if self.type == "list":
            for item in self.current_data:
                if item[key] == value:
                    return item
        return None

    def remove_item(self, key, value=None):
        if self.type == "list":
            for item in self.current_data:
                if item[key] == value:
                    self.current_data.remove(item)
                    self.save()
                    return
        elif self.type == "single":
            del self.current_data[key]
            self.save()


class DataStoreManager():
    def __init__(self) -> None:
        self.datastore_list = []

    def get_datastore_by_name(self, name):
        for datastore in self.datastore_list:
            if datastore.name == name:
                return datastore
        return None

    def get_data_list(self, name):
        datastore = self.get_datastore_by_name(name)
        if datastore is None:
            datastore = DataStore(name)
            self.datastore_list.append(datastore)
        return datastore.get()

    def save_data_list(self, name, data_list):
        datastore = self.get_datastore_by_name(name)
        if datastore is None:
            datastore = DataStore(name)
            self.datastore_list.append(datastore)
        datastore.set_all(data_list)
        return True

    def add_data_to_list(self, name, data):
        datastore = self.get_datastore_by_name(name)
        if datastore is not None:
            datastore.add(data)
        else:
            print("datastore not found:", name)

    def remove_data_from_list(self, name, key, value):
        datastore = self.get_datastore_by_name(name)
        if datastore is not None:
            datastore.remove_item(key, value)
        else:
            print("datastore not found:", name)


manager = DataStoreManager()
