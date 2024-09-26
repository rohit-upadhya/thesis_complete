import json
import os
import random

def load_json(file):
    with open(file,"r") as stream:
        json_data = json.load(stream)
    # print(json_data)
    return json_data
    pass

if __name__=="__main__":
    output_data_path = "output/russian/jsons/"
    files = []
    for (dirpath, dirnames, filenames) in os.walk(output_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    total_usable = 0
    for file in files:
        print(file)
        count = 0
        count_headings = 0
        count_usable = 0
        json_data = load_json(file)
        usable = []
        for entry in json_data:
            try:
                if len(entry['case_name'])>0 and len(entry['relevant_paragrpahs'][0]) == 0:
                    # print(entry)
                    count += 1
                elif len(entry['case_name'])==0:
                    count_headings += 1
                elif len(entry["query"][0]) == 0:
                    count += 1
                else:
                    usable.append(entry)
                    count_usable += 1
            except Exception as e:
                print("issue with the datapoint. Skipping")
        total_usable += count_usable
        
        sample_size = int(len(usable) * 0.1)
        sample_size = max(sample_size, 10)
        sample_size = min (sample_size, len(usable))
        sampled_data = random.sample(usable, sample_size)
        for idx, sample in enumerate(sampled_data):
            sample["id"] = idx
        file_name = file.split("/")[-1].split(".json")[0]
        output_file =  os.path.join("output", "russian", "relevant_jsons", f"{file_name}_analysis.json")
        with open(output_file, 'w') as file:
            json.dump(usable, file, indent=4, ensure_ascii=False,)
        meta_data_file = os.path.join("output", "russian", "analysis", "metadata_analysis.txt")
        with open(meta_data_file, 'a+') as file:
            file.write(f"for {file_name} \t\t |incorrectly parsed : {count} \t | missing cases : {count_headings} \t | usable cases : {count_usable}. \n\n\n")
        print("incorrectly parsed", count)
        print("missing cases", count_headings)
        print("usable cases", count_usable)
        print()
        
    print("\n\n******************")
    print("total datapoints : ",total_usable)
    print("******************\n\n")