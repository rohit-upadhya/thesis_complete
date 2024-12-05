import json
import os
from typing import List, Dict, Any, Text
import math
from transformers import BertTokenizerFast
from statistics import mean

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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
    query_tokens = tokenizer.tokenize(", ".join(json_dict["query"]))
    relevant_paragraphs_tokens = [len(tokenizer.tokenize(" ".join(paragraph))) for paragraph in json_dict["relevant_paragrpahs"]]
    total_paragraphs_tokens = [len(tokenizer.tokenize(" ".join(paragraph))) for paragraph in json_dict["all_paragraphs"]]
    return relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len(query_tokens), relevant_paragraphs_tokens, total_paragraphs_tokens

def make_data_dictionary(relevant_paragraphs, total_paragraphs, case_link, percentage, case_name, query_tokens, len_query_tokens, file, relevant_paragraphs_tokens, total_paragraphs_tokens):
    return ({
        "case_name": case_name,
        "case_link": case_link,
        "relevant_paragraphs": relevant_paragraphs,
        "total_paragraphs": total_paragraphs,
        "percentage": percentage,
        "query_tokens": len_query_tokens,
        "file_name": file,
        "relevant_paragraphs_tokens": relevant_paragraphs_tokens,
        "total_paragraphs_tokens": total_paragraphs_tokens
    })

def dump_json(path, data, lang):
    with open(os.path.join(path,f"{lang}.json"), "w+") as file:
        json.dump(data, file, indent=4, ensure_ascii=False,)
    pass

def run_percentage(files, lang):
    meta_data_information: List[Dict[Text, Any]] = []
    max_percentage = 0
    min_percentage = math.inf
    max_paragraphs = 0
    min_paragraphs = math.inf

    total_percentages = []
    total_query_tokens = []
    total_para_tokens = []
    total_paragraph_counts = []

    for file in files:
        print(file)
        json_data = extract_output_jsons(file)
        file_meta_data: List[Dict[Text, Any]] = []
        file_total_percentage = 0

        for json_dict in json_data:
            (
                relevant_paragraphs,
                total_paragraphs,
                case_link,
                percentage,
                case_name,
                query_tokens,
                len_query_tokens,
                relevant_paragraphs_tokens,
                total_paragraphs_tokens,
            ) = find_number_of_docs(json_dict)

            total_percentages.append(percentage)
            total_query_tokens.append(len_query_tokens)
            total_para_tokens.extend(total_paragraphs_tokens)
            total_paragraph_counts.append(total_paragraphs)

            if max_percentage < percentage:
                max_percentage = percentage
            if min_percentage > percentage:
                min_percentage = percentage

            if max_paragraphs < total_paragraphs:
                max_paragraphs = total_paragraphs
            if min_paragraphs > total_paragraphs:
                min_paragraphs = total_paragraphs

            file_meta_data.append(
                make_data_dictionary(
                    relevant_paragraphs,
                    total_paragraphs,
                    case_link,
                    percentage,
                    case_name,
                    query_tokens,
                    len_query_tokens,
                    file,
                    relevant_paragraphs_tokens,
                    total_paragraphs_tokens,
                )
            )

        try:
            avg = mean(total_percentages)
        except ZeroDivisionError:
            avg = "NA"

        meta_data_information.append(
            {
                "file_name": os.path.basename(file),
                "average_percentage": avg,
                "file_meta_data_information": file_meta_data,
            }
        )

    output_data_path = f"/home/upadro/code/thesis/data_analysis/specifics/new_split"
    dump_json(output_data_path, meta_data_information, lang=lang)

    final_results = {
        "language": lang,
        "min_percentage": min_percentage,
        "max_percentage": max_percentage,
        "mean_percentage": mean(total_percentages),
        "max_paragraph": max_paragraphs,
        "min_paragraph": min_paragraphs,
        "mean_para": mean(total_paragraph_counts),
        "mean_query": mean(total_query_tokens),
        "mean_tokens": mean(total_para_tokens),
    }
    info_output = os.path.join(output_data_path, "info")
    dump_json(info_output, final_results, lang=f"{lang}_all_info")

    print("min ", min_percentage)
    print("max ", max_percentage)

def run_unique_number_queries(files, split):
    queries = []
    for file in files:
        json_data = extract_output_jsons(file)
        for json_dict in json_data:
            queries.append(json_dict["query"])
    unique_queries = set(tuple(query) for query in queries)
    json_data = {
        "number_of_q_d_pairs": len(queries),
        "number_of_unique_queries": len(unique_queries)
    }
    dump_json(path=f"/srv/upadro/data_analysis/unseen_queries/{split}/counts",data=json_data, lang="")

if __name__=="__main__":
    
    # languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    
    for language in languages:
        all_files = []
        splits = ["train_test_val", "val", "test", "unique_query"]
        for split in splits:
            folder = os.path.join("input/train_infer/", language, "new_split", split)
            for (dirpath, dirnames, filenames) in os.walk(folder):
                for filename in filenames:
                    if "json" in filename:
                        all_files.append(os.path.join(dirpath, filename))
        print(all_files[1], all_files[-1], all_files[int(len(all_files)/2)])
        run_percentage(all_files, lang=language)
    
    # train_data_path = "/srv/upadro/dataset/english/unseen_queries/train"
    # test_data_path = "/srv/upadro/dataset/english/unseen_queries/test"
    # unique_query_test_data_path = "/srv/upadro/dataset/english/unseen_queries/unique_query_test"
    # val_data_path = "/srv/upadro/dataset/english/unseen_queries/val"
    # train_test_val = "/srv/upadro/dataset/english/unseen_queries/train_test_val"
    # data_paths = [train_data_path, test_data_path, val_data_path, unique_query_test_data_path, train_test_val]
    # for input_data_path in data_paths:
    #     files = []
    #     for (dirpath, dirnames, filenames) in os.walk(input_data_path):
    #         for filename in filenames:
    #             if "json" in filename:
    #                 files.append(os.path.join(dirpath, filename))
    #     split = input_data_path.split("/")[-1]
    #     run_percentage(files, split)
    #     run_unique_number_queries(files, split)
    
    
    
    
    
        
        