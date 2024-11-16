from rank_bm25 import BM25Okapi
import os
import numpy as np
from src.models.vector_db.commons.input_loader import InputLoader
from tqdm import tqdm
import json

def load_all_input_from_dir(input_data_path, langugage):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(input_data_path):
        for filename in filenames:
            if "json" in filename:
                files.append(os.path.join(dirpath, filename))
    input_loader = InputLoader()
    total_inference_datapoints = []
    for file in files:
        individual_datapoints = input_loader.load_data(data_file=file)
        total_inference_datapoints.extend(individual_datapoints)  # type: ignore
    translatons_query = {}
    if langugage != 'english':
        translatons_query = input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{langugage}.json")
    
    for data_point in total_inference_datapoints:
        data_point["query_translated"] = []
        for query_item in data_point["query"]:
            if langugage != 'english':
                
                for idx, trdata_point in enumerate(translatons_query):
                    
                    if trdata_point["original"] == query_item:
                        data_point["query_translated"].append(trdata_point["translation"])
            else:
                data_point["query_translated"] = data_point["query"]
    
    print(total_inference_datapoints[0]["query_translated"], total_inference_datapoints[0]["query"])
    return total_inference_datapoints

def preprocess(text):
    
    return text.lower().split()

def bm25_similarity(datapoint):
    all_paragraphs = [" ".join(paragraph) for paragraph in datapoint["all_paragraphs"]]
    
    golden_keys = datapoint.get("paragraph_numbers", [])

    tokenized_paragraphs = [preprocess(paragraph) for paragraph in all_paragraphs]
    
    bm25 = BM25Okapi(tokenized_paragraphs)
    
    query = ", ".join(datapoint["query_translated"])
    
    tokenized_query = preprocess(query)
    
    scores = bm25.get_scores(tokenized_query)
    
    paragraph_scores = []
    for i, score in enumerate(scores):
        paragraph_scores.append((i + 1, all_paragraphs[i], score))
    
    sorted_paragraphs = sorted(paragraph_scores, key=lambda x: x[2], reverse=True)
    return sorted_paragraphs, golden_keys

def recall_at_k(actual, predicted, k):
    relevance = [1 if x in actual else 0 for x in predicted]
    r = np.asarray(relevance)[:k]
    return np.sum(r) / len(actual) if len(actual) > 0 else 0

def calculate_recall(data_points, results):
    percentages = [2, 5, 10]
    recall_scores = {}
    
    for percentage in percentages:
        r_at_percentage = []
        for idx, result in enumerate(results):
            actual_relevant = set(result['golden_keys'])
            
            predicted_relevant = [x[0] for x in result['ranked_paragraphs']]  
            k = max(1, len(predicted_relevant) * percentage // 100)  
            
            recall_score = recall_at_k(actual_relevant, predicted_relevant, k)
            r_at_percentage.append(recall_score)
        
        recall_scores[f"mean_recall_at_{percentage}_percentage"] = np.mean(np.asarray(r_at_percentage))
    
    return recall_scores

if __name__ == "__main__":
    input_folder = "input/inference_input/russian/unique_query_test"
    
    languages = ["english","french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    # languages = ["english"]
    files = ["test","unique_query"]
    for file in files:
        for language in tqdm(languages):
            input_folder = f"input/train_infer/{language}/new_split/{file}"
            data = load_all_input_from_dir(input_folder, language)
            results = []

            for datapoint in tqdm(data):
                similarities, golden_keys = bm25_similarity(datapoint)
                
                ranked_paragraph_numbers = [paragraph_number for paragraph_number, _, _ in similarities]
                
                results.append({
                    "ranked_paragraphs": similarities, 
                    "golden_keys": golden_keys
                })
                
                # print("Paragraph Numbers - ", ranked_paragraph_numbers)
                # print("Golden Keys  - ", golden_keys)
            
            recall_scores = calculate_recall(data, results)
            print(f"Recall: {recall_scores}")
            
            recall_data = {
                "model": "bm25"
            }
            recall_path = os.path.join('output/inference_outputs/new_splits',f'recall_bm25_{file}.json')
            if os.path.exists(recall_path):
                with open(recall_path, 'r') as json_file:
                    recall_data = json.load(json_file)
                    
            # with open(recall_path, 'r') as json_file:
            #     recall_data = json.load(json_file)

            recall_data[language] = recall_scores
            with open(recall_path, 'w+') as json_file:
                json.dump(recall_data, json_file, indent=4, ensure_ascii=False)
            print(recall_scores)
