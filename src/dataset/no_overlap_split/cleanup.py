import os
import json


def count_percentage(json_dict):
    try:
        relevant_paragraphs = len(json_dict['relevant_paragrpahs'])
        total_paragraphs = len(json_dict["all_paragraphs"])
        percentage = relevant_paragraphs/total_paragraphs
        percentage = round(percentage, 4)
        if percentage >=0.20:
            return False
        return True
    except:
        return False

def dump_json(file_name, data, language):
    with open(os.path.join(f"/srv/upadro/dataset/{language}/done_new", f"{file_name}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)
        
def count_total_tokens(json_dict):
    try:
        total_paragraphs_tokens = [len(" ".join(paragraphs).split()) for paragraphs in json_dict["all_paragraphs"]]
        if max(total_paragraphs_tokens) > 3500:
            return False
        return True
    except:
        return False        
def loads_json(file_name):
    with open(file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

if __name__=="__main__":
    
    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    
    for language in languages:
        input_data_path = f"/srv/upadro/dataset/{language}/done"
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        
        for file in files:
            json_data = loads_json(file)
            json_list = []
            id = 0
            for data in json_data:
                if count_percentage(data) and count_total_tokens(data):
                    data["id"] = id
                    json_list.append(data)
                    id+=1
            file_name = file.split("/")[-1].split(".json")[0].split("_analysis")[0]
            dump_json(file_name=file_name, data=json_list, language=language)
            print(file)
        