import json
import os

def load_json(file):
    with open(file,"r") as stream:
        json_data = json.load(stream)
    # print(json_data)
    return json_data
    pass

if __name__=="__main__":
    output_data_path = "output/turkish/"
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
        for entry in json_data:
            if len(entry['case_name'])>0 and len(entry['paragrpahs'][0]) == 0:
                
                # print(entry)
                count += 1
            elif len(entry['case_name'])==0:
                count_headings += 1
            
            else:
                count_usable += 1
        total_usable += count_usable
        print("incorrectly parsed", count)
        print("missing cases", count_headings)
        print("usable cases", count_usable)
        print()
        
    print("\n\n******************")
    print("total datapoints : ",total_usable)
    print("******************\n\n")