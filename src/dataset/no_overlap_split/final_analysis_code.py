import os
import json
import pandas as pd

def load_files_for_lang(input_data_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    return files

def create_language_dataframe(final_data):
    columns = [
        "Total unique queries", "Total Q-J pairs", "Number of paragraphs per judgement",
        "Percent of relevant paragraphs per Q-J pair", "Uniquer queries in unseen test",
        "Unseen test Q-J pairs", "Train", "Valid", "Seen Test"
    ]

    languages = ["Eng.", "Fre.", "Ita.", "Romn.", "Rus.", "Turk.", "Ukr."]

    data_dict = {lang: [] for lang in languages}

    for lang, entry in zip(languages, final_data):
        data_dict[lang].extend([
            entry["unique_queries_total"],
            entry["q_j_total"],
            entry["paragraph_per_judgement"],
            entry["percentage_relevant_paragraph_per_judgement"],
            entry["unique_query"]["unique_queries"],
            entry["unique_query"]["q_j"],
            entry["train_test_val"]["q_j"],
            entry["val"]["q_j"],
            entry["test"]["q_j"]
        ])

    df = pd.DataFrame(data_dict, index=columns)
    return df

def loads_json(load_file_name):
    with open(load_file_name, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return data

if __name__ == "__main__":

    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]

    final_data = []
    for language in languages:

        splits = ["train_test_val", "val", "test", "unique_query"]

        info = {
            "language": language,
            "unique_queries_total": 0,
            "q_j_total": 0,
            "percentage_relevant_paragraph_per_judgement": 0,
            "paragraph_per_judgement": 0,
            
        }
        total_data = 0
        paragraphs = 0
        percentage = 0
        for split in splits:
            
            folder = os.path.join("input/train_infer/", language, "new_split", split)
            files = load_files_for_lang(folder)
            info[split] = {
                    "unique_queries": 0,
                    "q_j": 0,
                }
            for file in files:
                data = loads_json(file)
                total_qj = set()
                unique_queries = set()
                total_data += len(data)
                
                for item in data:
                    qj_str = f"{', '.join(item['query'])}_{item['link']}"
                    unique_query_str = ', '.join(item['query'])
                    
                    total_qj.add(qj_str)
                    unique_queries.add(unique_query_str)
                    paragraphs += len(item["all_paragraphs"])
                    # print(len(item["paragraph_numbers"]))
                    percentage += len(item["paragraph_numbers"]) / len(item["all_paragraphs"])
                    # print(percentage)
                info["unique_queries_total"] += len(unique_queries)
                info["q_j_total"] += len(total_qj)
                
                info[split]["unique_queries"] += len(unique_queries)
                info[split]["q_j"] += len(total_qj)
        
        info["percentage_relevant_paragraph_per_judgement"] = percentage / total_data
        info["paragraph_per_judgement"] = paragraphs / total_data
                
        final_data.append(info)
        print(f"{language} : Done!")
    # Save the final data to JSON
    output_path = "output/new_split_info/language_data.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)

    print(f"Data successfully saved to {output_path}")
    
    df = create_language_dataframe(final_data)
    
    # Save the DataFrame to a CSV file
    output_csv_path = "output/new_split_info/language_data.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=True)

    print(f"CSV file successfully saved to {output_csv_path}")
