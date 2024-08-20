import json
import os
from typing import List, Dict, Any, Text
import math
# file_path = os.path.abspath(__file__)
# base_path = os.path.join(file_path.split("thesis")[0],"thesis")

def extract_output_jsons(file_name):
    with open(file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

def find_number_of_docs(json_dict):
    relevant_paragraphs = len(json_dict['relevant_paragrpahs'])
    total_paragraphs = len(json_dict["all_paragraphs"])
    case_link = json_dict['link']
    case_name = json_dict['case_name']
    percentage = relevant_paragraphs/total_paragraphs
    percentage = round(percentage, 4)
    query_tokens = (" ".join(json_dict["query"])).split()
    return relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len(query_tokens)

def make_data_dictionary(relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len_query_tokens, file):
    return ({
        "case_name": case_name,
        "case_link": case_link,
        "relevant_paragraphs": relevant_paragraphs,
        "total_paragraphs": total_paragraphs,
        "percentage": percentage,
        "query_tokens": len_query_tokens,
        "file_name": file
    })

def dump_json(path, data):
    with open(os.path.join(path,"russian.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)
    pass


if __name__=="__main__":
    input_data_path = "output/russian/relevant_jsons"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    meta_data_information: List[Dict[Text,Any]] = []
    max_ = 0
    min_ = math.inf
    for file in files:
        print(file)
        json_data = extract_output_jsons(file)
        file_meta_data: List[Dict[Text,Any]] = []
        file_total_percentage = 0
        for json_dict in json_data:
            relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len_query_tokens = find_number_of_docs(json_dict)
            file_total_percentage += percentage
            if max_ < percentage:
                max_ = percentage
            if min_ > percentage:
                min_ = percentage
            file_meta_data.append(make_data_dictionary(relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len_query_tokens, file))
        
        try:
            avg = file_total_percentage/len(file_meta_data)
        except:
            avg = "NA"
        meta_data_information.append({
            "file_name": os.path.basename(file),
            "average_percentage": avg,
            "file_meta_data_information": file_meta_data
        })
        
    output_data_path = "data_analysis"
    dump_json(output_data_path,meta_data_information)
    print("min ",min_)
    print("max ",max_)
        
        