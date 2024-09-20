from typing import Text, List, Dict, Any, Optional
import numpy as np
import os
import json

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.encoder import Encoder
from src.models.vector_db.inference.faiss_vector_db import FaissVectorDB

class Inference:
    def __init__(
            self, 
            inference_folder: Optional[Text] = None, 
            inference_datapoints: Optional[List[Dict[str, Any]]] = None, 
            bulk_inference: bool = False, 
            number_of_relevant_paragraphs: int = 5,
            number_of_relevant_percentage: float = 2.0,
            device: str = "cpu"
        ):
        if bulk_inference:
            inference_datapoints = self._load_all_input_from_dir(inference_folder)
        self._format_input(inference_datapoints)
        self.number_of_relevant_paragraphs = number_of_relevant_paragraphs
        self.number_of_relevant_percentage = float(float(number_of_relevant_percentage)/100)
        self.encoder = Encoder(device=device)
        self.faiss = FaissVectorDB(device=device)
        self.all_paragraph_encodings = []
        self.all_unique_keys = []
        self.query_paragraph_mapping = {}
    
    def calculate_recall(self, results):
        total_relevant = 0
        total_retrieved_relevant = 0

        for idx, result in enumerate(results):
            query_data = self.data_points[idx]
            actual_relevant = set(query_data.get('paragraph_numbers', []))
            predicted_relevant = set(result['relevant_paragraphs'])
            retrieved_relevant = len(actual_relevant.intersection(predicted_relevant))
            total_relevant += len(actual_relevant)
            total_retrieved_relevant += retrieved_relevant

        recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
        return recall
    
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
            print(len(individual_datapoints)) # type: ignore
            total_inference_datapoints.extend(individual_datapoints) # type: ignore
        return total_inference_datapoints
        
    
    def _format_input(self, inference_datapoints):
        self.data_points = []
        for idx, data in enumerate(inference_datapoints):
            key = data["link"]
            unique_key = f"datapoint_{key}"
            self.data_points.append(
                {
                    "query": ", ".join(data["query"]),
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i}" for i in range(len(data["all_paragraphs"]))],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "link": key,
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
    
    def _encode_all_paragraphs(self):
        index_counter = 0
        paragraph_set = set()
        for idx, points in enumerate(self.data_points):
            if points["link"] not in paragraph_set:
                encoded_paragraphs = self.encoder.encode(points["all_paragraphs"]).cpu().numpy()
                self.all_paragraph_encodings.append(encoded_paragraphs)
                self.all_unique_keys.extend(points["unique_keys"])

                self.query_paragraph_mapping[idx] = list(range(index_counter, index_counter + len(points["all_paragraphs"])))
                index_counter += len(points["all_paragraphs"])
                paragraph_set.add(points["link"])

        all_encodings = np.vstack(self.all_paragraph_encodings)
        self.faiss.build_index(all_encodings, self.all_unique_keys)

    def _encode_query(self, query: Text):
        return self.encoder.encode([query]).cpu().numpy()
    
    def main(self):
        self._encode_all_paragraphs()
        results = []
        
        for idx, points in enumerate(self.data_points):
            query_encodings = self._encode_query(points['query'])
            number_of_relevant_paragraphs = int(points["length_of_all_paragraphs"] * self.number_of_relevant_percentage)
            relevant_paragraph_keys = self.faiss.perform_search(
                query=query_encodings,
                number_of_results=number_of_relevant_paragraphs,
                datapoint = points
            )
            results.append({
                "query": points["query"],
                "link": points["link"],
                "relevant_paragraphs": [int(paragraph.split("_")[-1])+1 for paragraph in relevant_paragraph_keys],
            })
        recall = self.calculate_recall(results)
        print(f"Recall: {recall:.4f}")
        print(results)
        with open(os.path.join('output/inference_outputs','results.json'), 'w') as json_file:
            json.dump(results, json_file, indent=4, ensure_ascii=False)
        
        return results

if __name__ == "__main__":
    inference_folder = "/home/upadro/code/thesis/src/models/single_datapoints/common"
    bulk_inference = True
    inference = Inference(
        inference_folder=inference_folder, 
        bulk_inference=bulk_inference,
        number_of_relevant_percentage=2,
        device='cuda:0'
    )
    inference.main()
    