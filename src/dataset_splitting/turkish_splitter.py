import os
import json
import random
import math

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def dump_json(base_path, unique_file_name, data):
    with open(os.path.join(base_path, f"{unique_file_name}.json"), "w+") as file:
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
        
        query_dict = {}
        
        for item in json_data:
            if " ".join(item["query"]) not in query_dict.keys():
                query_dict[" ".join(item["query"])] = 1
            else:
                query_dict[" ".join(item["query"])] += 1
        sorted_dict = dict(sorted(query_dict.items(), key=lambda item: item[1]))
        sorted_keys = [key for key, value in sorted_dict.items()]
        
        top_20_percent = int(math.ceil(len(sorted_keys) * 0.2))
        top_20_list = sorted_keys[:top_20_percent]
        remaining_80_list = sorted_keys[top_20_percent:]
        random.shuffle(top_20_list)
        half_20_count = int(top_20_percent * 0.5)
        half_20_count = max(1, half_20_count)
        
        test_queries = top_20_list[:half_20_count]
        train_queries = remaining_80_list + top_20_list[half_20_count:]
        
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
        dump_json(base_path=train_base_path, unique_file_name=file_name_train, data=train_data)
        dump_json(base_path=test_base_path, unique_file_name=file_name_test, data=test_data)
        print(file)
        
        
        