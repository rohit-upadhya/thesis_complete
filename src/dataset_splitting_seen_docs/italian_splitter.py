import os
import json
import random
import math
import logging

logging.basicConfig(filename='italian_splitter.log', level=logging.INFO)

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def validate_split(train_data, test_data):
    train_links = set(item['link'] for item in train_data)
    test_links = set(item['link'] for item in test_data)
    
    if train_links.intersection(test_links):
        print("Warning: Overlap detected between train and test sets")
        logging.warning("Overlap detected between train and test sets")
    else:
        print("Validation passed: No overlap between train and test sets")
        logging.info("Validation passed: No overlap between train and test sets")

def dump_json(base_path, unique_file_name, data):
    with open(os.path.join(base_path, f"{unique_file_name}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)

if __name__=="__main__":
    input_data_path = "output/italian/done"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    train_base_path = "/srv/upadro/dataset/italian/unseen_docs/train"
    test_base_path = "/srv/upadro/dataset/italian/unseen_docs/test"
    
    doc_set = set()
    for file in files:
        json_data = loads_json(file)
        for data_point in json_data:
            if "link" in data_point:
                doc_set.add(data_point["link"])
    
    doc_list = list(doc_set)
    random.shuffle(doc_list)
    
    top_10_percent = doc_list[:math.ceil(len(doc_list)*0.1)]
    rest_90_percent = doc_list[math.ceil(len(doc_list)*0.1):]
    print(len(rest_90_percent))

    total_train_data = 0
    total_test_data = 0

    for file in files:
        json_data = loads_json(file)
        
        test_data = []
        train_data = []
        for item in json_data:
            link = item["link"]
            print(link)
            if link in rest_90_percent:
                train_data.append(item)
            elif link in top_10_percent:
                print("here")
                test_data.append(item)
        file_name = file.split("/")[-1].split("_relevant.json")[0]
        file_name_train = f"{file_name}_train"
        file_name_test = f"{file_name}_test"
        dump_json(base_path=train_base_path, unique_file_name=file_name_train, data=train_data)
        dump_json(base_path=test_base_path, unique_file_name=file_name_test, data=test_data)
        print(file)
        
        # Add validation
        validate_split(train_data, test_data)
        
        # Add error handling and logging
        if not train_data and not test_data:
            print(f"Warning: No data was split for file {file}")
            logging.warning(f"No data was split for file {file}")
        else:
            logging.info(f"Processed file: {file}")
            logging.info(f"Train data points: {len(train_data)}")
            logging.info(f"Test data points: {len(test_data)}")
        
        total_train_data += len(train_data)
        total_test_data += len(test_data)

    # Add summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total documents: {len(doc_list)}")
    print(f"Train documents: {len(rest_90_percent)}")
    print(f"Test documents: {len(top_10_percent)}")
    print(f"Total train data points: {total_train_data}")
    print(f"Total test data points: {total_test_data}")
    logging.info(f"Total documents: {len(doc_list)}")
    logging.info(f"Train documents: {len(rest_90_percent)}")
    logging.info(f"Test documents: {len(top_10_percent)}")
    logging.info(f"Total train data points: {total_train_data}")
    logging.info(f"Total test data points: {total_test_data}")
