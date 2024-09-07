import os
import json
import random

def loads_json(file_name):
    with open(file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def dump_json(base_path, file_name, data):
    with open(os.path.join(base_path, f"{file_name}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)
        
if __name__=="__main__":
    input_data_path = "output/turkish/done"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    train_base_path = "/srv/upadro/dataset/turkish/train"
    test_base_path = "/srv/upadro/dataset/turkish/test"
    for file in files:
        json_data = loads_json(file)
        query_set = set()
        for item in json_data:
            query_str = " ".join(item["query"])
            query_set.add(query_str)
            
        query_list = list(query_set)
        random.shuffle(query_list)
        ten_percent_length = int(0.1 * len(query_list))
        test_queries = query_list[:ten_percent_length]
        train_queries = query_list[ten_percent_length:]
        
        test_data = []
        train_data = []
        for item in json_data:
            query_str = " ".join(item["query"])
            if query_str in train_queries:
                train_data.append(item)
            elif query_str in test_queries:
                test_data.append(item)
        file_name = file.split("/")[-1].split("_relevant.json")[0]
        file_name_train = f"{file_name}_train"
        file_name_test = f"{file_name}_test"
        dump_json(base_path=train_base_path, file_name=file_name, data=train_data)
        dump_json(base_path=test_base_path, file_name=file_name, data=test_data)
        print(file)
        
        
        