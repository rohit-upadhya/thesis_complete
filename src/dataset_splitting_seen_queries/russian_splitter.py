import os
import json
import random
import math
from copy import deepcopy

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def dump_json(base_path, unique_file_name, data):
    with open(os.path.join(base_path, f"{unique_file_name}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)
        
if __name__=="__main__":
    input_data_path = "output/russian/done"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    train_base_path = "/srv/upadro/dataset/russian/unseen_queries/train"
    train_test_val_base_path = "/srv/upadro/dataset/russian/unseen_queries/train_test_val"
    val_base_path = "/srv/upadro/dataset/russian/unseen_queries/val"
    test_base_path = "/srv/upadro/dataset/russian/unseen_queries/test"
    unique_query_test_base_path = "/srv/upadro/dataset/russian/unseen_queries/unique_query_test"
    for file in files:
        json_data = loads_json(file)
        query_set = set()
        
        for item in json_data:
            query_set.add(" ".join(item["query"]))
        
        query_list = list(query_set)
        rerun_split = True
        while(rerun_split):
            query_list_for_split = deepcopy(query_list)
            random.shuffle(query_list_for_split)
            
            test_queries = query_list_for_split[:int(math.ceil(len(query_list_for_split) * 0.1))]
            train_queries = query_list_for_split[int(math.ceil(len(query_list_for_split) * 0.1)):]
            unseen_queries_test_data = []
            train_data = []
            for item in json_data:
                query_str = " ".join(item["query"])
                if query_str in train_queries:
                    train_data.append(item)
                elif query_str in test_queries:
                    unseen_queries_test_data.append(item)
            if (len(unseen_queries_test_data)/len(json_data)) > 0.15:
                print((len(unseen_queries_test_data)/len(json_data)), file)
            if 0.08<(len(unseen_queries_test_data)/len(json_data))<0.20 or len(json_data) < 30:
                rerun_split = False
        unique_query_test_data = unseen_queries_test_data
        
        train_val_test_data = deepcopy(train_data)
        random.shuffle(train_val_test_data)
        test_size = val_size = int(0.1 * len(train_val_test_data))
        new_test = [f'{str(item["query"])} {item["case_name"]}' for item in train_val_test_data[:test_size]]
        val = [f'{str(item["query"])} {item["case_name"]}' for item in train_val_test_data[test_size:test_size+val_size]] 
        final_train = [f'{str(item["query"])} {item["case_name"]}' for item in train_val_test_data[test_size+val_size:]] 
        
        new_test_data = []
        val_data = []
        final_train_data = []
        
        for item in train_data:
            q_d = f'{str(item["query"])} {item["case_name"]}'
            if q_d in new_test:
                new_test_data.append(item)
            elif q_d in val:
                val_data.append(item)
            else:
                final_train_data.append(item)
        file_name = file.split("/")[-1].split("_relevant.json")[0]
        file_name_train_val_test = f"{file_name}_train_test_val"
        file_name_train = f"{file_name}_train"
        file_name_val = f"{file_name}_val"
        file_name_test = f"{file_name}_test"
        file_name_unique_query_test = f"{file_name}_unique_query_test"
        dump_json(base_path=train_test_val_base_path, unique_file_name=file_name_train, data=train_data)
        dump_json(base_path=train_base_path, unique_file_name=file_name_train, data=final_train_data)
        dump_json(base_path=val_base_path, unique_file_name=file_name_val, data=val_data)
        dump_json(base_path=test_base_path, unique_file_name=file_name_test, data=new_test_data)
        dump_json(base_path=unique_query_test_base_path, unique_file_name=file_name_unique_query_test, data=unique_query_test_data)
        print(file)