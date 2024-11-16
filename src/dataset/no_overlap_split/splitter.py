import os
import json
from copy import deepcopy
import random, math

def load_files_for_lang(input_data_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    return files

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def dump_json(base_path, unique_file_name, data):
    with open(os.path.join(base_path, f"{unique_file_name}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)

if __name__=="__main__":
    
    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    
    for language in languages:
        print(f"{'*'*10} {language} {'*'*10}")
        train_test_val_base_path = f"input/train_infer/{language}/new_split/train_test_val"
        val_base_path = f"input/train_infer/{language}/new_split/val"
        test_base_path = f"input/train_infer/{language}/new_split/test"
        
        train_folder = f"input/train_infer/{language}/new_split/train"
        
        files = load_files_for_lang(train_folder)
        
        for file in files:
            print(f"{file}")
            json_data = loads_json(file)
            
            train_val_test_data = deepcopy(json_data)
            
            random.shuffle(train_val_test_data)
            
            test_size = val_size = int(math.ceil(0.1 * len(train_val_test_data)))
            
            new_test = [f'{str(item["query"])} {item["link"]}' for item in train_val_test_data[:test_size]]
            val = [f'{str(item["query"])} {item["link"]}' for item in train_val_test_data[test_size:test_size+val_size]] 
            final_train = [f'{str(item["query"])} {item["link"]}' for item in train_val_test_data[test_size+val_size:]] 
            
            new_test_data = []
            val_data = []
            final_train_data = []
            
            for item in json_data:
                q_d = f'{str(item["query"])} {item["link"]}'
                if q_d in new_test:
                    new_test_data.append(item)
                elif q_d in val:
                    val_data.append(item)
                else:
                    final_train_data.append(item)
            file_name = file.split("/")[-1].split("_relevant.json")[0]
            file_name_train_val_test = f"{file_name}_train"
            file_name_val = f"{file_name}_val"
            file_name_test = f"{file_name}_test"
            
            dump_json(base_path=train_test_val_base_path, unique_file_name=file_name_train_val_test, data=final_train_data)
            dump_json(base_path=val_base_path, unique_file_name=file_name_val, data=val_data)
            dump_json(base_path=test_base_path, unique_file_name=file_name_test, data=new_test_data)