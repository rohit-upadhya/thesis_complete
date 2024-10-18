from typing import Text, List, Dict, Any, Optional
import numpy as np # type: ignore
import os
import json
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.dpr_inference.encoder_dpr import Encoder
from src.models.vector_db.inference.dpr_inference.faiss_vector_db_dpr import FaissVectorDB

class Inference:
    def __init__(
            self, 
            inference_folder: Optional[Text] = None, 
            inference_datapoints: Optional[List[Dict[str, Any]]] = None, 
            bulk_inference: bool = False,
            use_translations: bool = False,
            device: str = "cpu",
            language: str = "english"
        ):
        self.input_loader = InputLoader()
        self.file_base_name = language
        self.use_translations = use_translations
        self.language = language
        if bulk_inference:
            inference_datapoints = self._load_all_input_from_dir(inference_folder)
            self.file_base_name = inference_folder.split("/")[-2] # type: ignore
        self._format_input(inference_datapoints)
        self.encoder = Encoder(device=device, model_type="dpr")
        self.faiss = FaissVectorDB(
            # device=device
            )
        self.all_paragraph_encodings = []
        self.all_unique_keys = []
        self.query_paragraph_mapping = {}
        
    
    
    def _load_all_input_from_dir(self, input_data_path):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(input_data_path):
            for filename in filenames:
                if "json" in filename:
                    files.append(os.path.join(dirpath, filename))
        total_inference_datapoints = []
        for file in files:
            individual_datapoints = self.input_loader.load_data(data_file=file)
            total_inference_datapoints.extend(individual_datapoints) # type: ignore
        return total_inference_datapoints
        
    
    def _format_input(self, inference_datapoints):
        self.data_points = []
        
        translatons_query = {}
        if self.language != 'english' and self.use_translations:
            translatons_query = self.input_loader.load_data(data_file=f"output/translation_outputs/query_translations_{self.language}.json")
        for idx, data in enumerate(inference_datapoints):
            key = data["link"]
            final_query = []
            if self.use_translations and self.language != 'english':
                for query_item in data["query"]:
                    for trdata_point in translatons_query:
                        if trdata_point["original"] == query_item:
                            final_query.append(trdata_point["translation"])
            else:
                final_query = data["query"]
            
            unique_key = f"datapoint_{key}"
            self.data_points.append(
                {
                    "query": ", ".join(final_query),
                    "all_paragraphs": ["\n".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i+1}" for i in range(len(data["all_paragraphs"]))],
                    "paragraph_numbers": data.get("paragraph_numbers", []),
                    "link": key,
                    "length_of_all_paragraphs": len(data["all_paragraphs"])
                }
            )
        print(self.data_points[0]["query"])
        print(inference_datapoints[0]["query"])
    def _encode_all_paragraphs(self, batch_size=128):
        index_counter = 0
        paragraph_set = set()
        for idx, points in tqdm(enumerate(self.data_points)):
            if points["link"] not in paragraph_set:
                all_paragraphs = points["all_paragraphs"]
                num_paragraphs = len(all_paragraphs)
                encoded_paragraphs = []

                for start_idx in range(0, num_paragraphs, batch_size):
                    end_idx = min(start_idx + batch_size, num_paragraphs)
                    batch_paragraphs = all_paragraphs[start_idx:end_idx]
                    
                    encoded_batch = self.encoder.encode(sentences=batch_paragraphs, is_query=False).cpu().numpy()
                    encoded_paragraphs.append(encoded_batch)

                encoded_paragraphs = np.vstack(encoded_paragraphs)
                
                self.all_paragraph_encodings.append(encoded_paragraphs)
                self.all_unique_keys.extend(points["unique_keys"])
                self.query_paragraph_mapping[idx] = list(range(index_counter, index_counter + len(all_paragraphs)))
                index_counter += len(all_paragraphs)
                paragraph_set.add(points["link"])
                
        all_encodings = np.vstack(self.all_paragraph_encodings)
        self.faiss.build_index(all_encodings, self.all_unique_keys)

    def _encode_query(self, query: Text):
        return self.encoder.encode(sentences=[query], is_query=True).cpu().numpy()
    
    
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
        
        for idx, points in enumerate(self.data_points):
            query_encodings = self._encode_query(points['query'])
            number_of_relevant_paragraph_2 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.02)), 1)
            number_of_relevant_paragraph_5 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.05)), 1)
            number_of_relevant_paragraph_10 = max(int(math.ceil(points["length_of_all_paragraphs"] * 0.1)), 1)            
            relevant_paragraph_keys = self.faiss.perform_search(
                query=query_encodings,
                datapoint = points,
            )
            results.append({
                "query": points["query"],
                "link": points["link"],
                "relevant_paragraphs": [int(paragraph.split("_")[-1]) for paragraph in relevant_paragraph_keys],
                "no_2": number_of_relevant_paragraph_2,
                "no_5": number_of_relevant_paragraph_5,
                "no_10": number_of_relevant_paragraph_10
            })
            
        recall = self.calculate_recall(results)
        print(f"Recall: {recall}")
        
        recall_data = {}
        recall_path = os.path.join('output/inference_outputs',f'recall_dpr.json')
        if os.path.exists(recall_path):
            with open(recall_path, 'r') as json_file:
                recall_data = json.load(json_file)
    
        recall_data[self.file_base_name] = recall
        recall_data["model"] = "dpr"
        with open(recall_path, 'w+') as json_file:
            json.dump(recall_data, json_file, indent=4, ensure_ascii=False)
        return results

if __name__ == "__main__":
    languages = ["english", "french", "italian", "romanian", "russian", "turkish", "ukrainian"]
    # languages = ["english"]
    for language in tqdm(languages, desc="Processing Languages"):
        print(f"Processing language: {language}")
        
        inference_folder = f"input/inference_input/{language}/unique_query_test"
        # inference_folder = "src/models/single_datapoints/common"
        bulk_inference = True
        use_translations = False
        inference = Inference(
            inference_folder=inference_folder, 
            bulk_inference=bulk_inference,
            use_translations=use_translations,
            device='cuda:0',
            language=language
        )
        inference.main()
        
        print(f"Completed processing for language: {language}")
        print("*" * 40)
    print("Inference process completed for all languages.")
    