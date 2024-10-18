from typing import Text, List, Dict, Any, Optional
import numpy as np  # type: ignore
import os
import json
import math
from tqdm import tqdm

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.test_faiss.encoder import Encoder
from src.models.vector_db.inference.test_faiss.faiss_vector_db import FaissVectorDB

class Inference:
    def __init__(
            self, 
            inference_folder: Optional[Text] = None, 
            inference_datapoints: Optional[List[Dict[str, Any]]] = None, 
            bulk_inference: bool = False,
            device: str = "cpu"
        ):
        self.file_base_name = "english"
        self.device_string = device
        if bulk_inference:
            inference_datapoints = self._load_all_input_from_dir(inference_folder)
            self.file_base_name = inference_folder.split("/")[-2]  # type: ignore
        self._format_input(inference_datapoints)
        self.encoder = Encoder(device=device)
        self.query_paragraph_mapping = {}
        self.faiss_db_mapping = {} 
    
    def _load_all_input_from_dir(self, input_data_path):
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
        return total_inference_datapoints
        
    def _format_input(self, inference_datapoints):
        self.data_points = []
        for idx, data in enumerate(inference_datapoints):
            self.data_points.append(
                {
                    "query": ", ".join(data["query"]),
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "link": data["link"],
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
    
    def _encode_all_paragraphs(self, batch_size=128):
        for idx, points in tqdm(enumerate(self.data_points)):
            faiss_db = FaissVectorDB()
            all_paragraphs = points["all_paragraphs"]
            num_paragraphs = len(all_paragraphs)

            all_encodings = []
            
            for start_idx in range(0, num_paragraphs, batch_size):
                end_idx = min(start_idx + batch_size, num_paragraphs)
                batch_paragraphs = all_paragraphs[start_idx:end_idx]

                encoded_batch = self.encoder.encode(batch_paragraphs).cpu().numpy()

                all_encodings.append(encoded_batch)

            all_encodings = np.vstack(all_encodings)

            self.query_paragraph_mapping[idx] = list(range(num_paragraphs))
            
            faiss_db.build_index(all_encodings)

            self.faiss_db_mapping[points["link"]] = faiss_db

        print("Encoding and indexing done for all data points!")


    
    def _encode_query(self, query: Text):
        return self.encoder.encode([query]).cpu().numpy()
    
    def recall_at_k(self, actual, predicted, k):
        relevance = [1 if x in actual else 0 for x in predicted]
        r = np.asarray(relevance)[:k]
        return np.sum(r) / len(actual) if len(actual) > 0 else 0

    def calculate_recall(self, results):
        percentages = [2, 5, 10]
        recall = {}
        
        for percentage in percentages:
            r_at_percentage = []
            for idx, result in enumerate(results):
                query_data = self.data_points[idx]
                actual_relevant = set(query_data.get('paragraph_numbers', []))
                
                predicted_relevant = result[f'relevant_paragraphs']
                k = result[f"no_{percentage}"]
                recall_score = self.recall_at_k(actual_relevant, predicted_relevant, k)
                r_at_percentage.append(recall_score)
            
            recall[f"mean_recall_at_{percentage}_percentage"] = np.mean(np.asarray(r_at_percentage))
        
        return recall

    def main(self):
        self._encode_all_paragraphs()
        results = []
        
        for points in tqdm(self.data_points):
            query_encodings = self._encode_query(points['query'])
            number_of_relevant_paragraph_2 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.02)), 1)
            number_of_relevant_paragraph_5 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.05)), 1)
            number_of_relevant_paragraph_10 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.1)), 1)
            
            faiss_db = self.faiss_db_mapping[points["link"]]
            relevant_paragraph_keys = faiss_db.perform_search(
                query=query_encodings,
                datapoint=points,
            )
            results.append({
                "query": points["query"],
                "link": points["link"],
                "relevant_paragraphs": [idx + 1 for idx in relevant_paragraph_keys],
                "no_2": number_of_relevant_paragraph_2,
                "no_5": number_of_relevant_paragraph_5,
                "no_10": number_of_relevant_paragraph_10
            })
            
        recall = self.calculate_recall(results)
        print(f"Recall: {recall}")
        # with open(os.path.join('output/inference_outputs', f'{self.file_base_name}_results.json'), 'w+') as json_file:
        #     json.dump(results, json_file, indent=4, ensure_ascii=False)
        
        recall_data = {}
        recall_path = os.path.join('output/inference_outputs',f'recall_english.json')
        if os.path.exists(recall_path):
            with open(recall_path, 'r') as json_file:
                recall_data = json.load(json_file)
    
        recall_data[self.file_base_name] = recall
        recall_data["model"] = self.encoder.model_name
        with open(recall_path, 'w+') as json_file:
            json.dump(recall_data, json_file, indent=4, ensure_ascii=False)
        return results

if __name__ == "__main__":
    inference_folder = "input/inference_input/english/unique_query_test"
    # inference_folder = "input/inference_input/ukrainian/unique_query_test"
    # inference_folder = "src/models/single_datapoints/common"
    # languages = ["french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    languages = ["english"]
    for language in tqdm(languages):
        print(language)
        inference_folder = f"input/inference_input/{language}/unique_query_test"
        bulk_inference = True
        inference = Inference(
            inference_folder=inference_folder, 
            bulk_inference=bulk_inference,
            device='cuda:1'
        )
        inference.main()
        print("*"*40)