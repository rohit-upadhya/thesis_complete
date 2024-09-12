from typing import Text, List, Dict, Any
import numpy as np

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.encoder import Encoder
from src.models.vector_db.inference.faiss_vector_db import FaissVectorDB

class Inference:
    def __init__(
            self, 
            inference_file: Text = None, 
            inference_datapoints: List[Dict[str, Any]] = None, 
            bulk_inference: bool = False, 
            number_of_relevant_paragraphs: int = 5,
            device: str = "cpu"
        ):
        input_loader = InputLoader()
        if bulk_inference:
            inference_datapoints = input_loader.load_data(inference_file)
        self._format_input(inference_datapoints)
        self.number_of_relevant_paragraphs = number_of_relevant_paragraphs
        self.encoder = Encoder(device=device)
        self.faiss = FaissVectorDB(device=device)
        self.all_paragraph_encodings = []
        self.all_unique_keys = []
        self.query_paragraph_mapping = {}
    
    def _format_input(self, inference_datapoints):
        self.data_points = []
        for idx, data in enumerate(inference_datapoints):
            unique_key = f"datapoint_{idx}"
            self.data_points.append(
                {
                    "query": " ".join(data["query"]),
                    "all_paragraphs": [" ".join(paras) for paras in data["all_paragraphs"]],
                    "unique_keys": [f"{unique_key}_para_{i}" for i in range(len(data["all_paragraphs"]))]
                }
            )
    
    def _encode_all_paragraphs(self):
        index_counter = 0
        for idx, points in enumerate(self.data_points):
            encoded_paragraphs = self.encoder.encode(points["all_paragraphs"]).cpu().numpy()
            self.all_paragraph_encodings.append(encoded_paragraphs)
            self.all_unique_keys.extend(points["unique_keys"])

            self.query_paragraph_mapping[idx] = list(range(index_counter, index_counter + len(points["all_paragraphs"])))
            index_counter += len(points["all_paragraphs"])

        all_encodings = np.vstack(self.all_paragraph_encodings)
        self.faiss.build_index(all_encodings, self.all_unique_keys)

    def _encode_query(self, query: Text):
        return self.encoder.encode([query]).cpu().numpy()

    # def _encode_queries_paragraphs(self, query: Text, all_paragraphs: List[Text], query_encoder: Encoder, parar_encoder: Encoder):
    #     query_encodings = query_encoder.encode([query]).cpu().numpy()  # Keep in NumPy
    #     all_paragraph_encodings = parar_encoder.encode(all_paragraphs).cpu().numpy()  # Convert to NumPy for FAISS
    #     return query_encodings, all_paragraph_encodings
    
    def main(self):
        self._encode_all_paragraphs()
        results = []
        
        for idx, points in enumerate(self.data_points):
            query_encodings = self._encode_query(points['query'])

            relevant_indices = self.query_paragraph_mapping[idx]

            distances, relevant_paragraph_keys = self.faiss.perform_search_with_indices(
                query=query_encodings,
                subset_indices=relevant_indices,
                number_of_relevant_paragraphs=self.number_of_relevant_paragraphs
            )
            results.append({
                "query": points["query"],
                "relevant_paragraphs": [int(paragraph.split("_")[-1])+1 for paragraph in relevant_paragraph_keys],
                "distances": distances
            })
            
        print(results)

if __name__ == "__main__":
    inference_file = "/home/upadro/code/thesis/src/models/single_datapoints/common/test.json"
    bulk_inference = True
    inference = Inference(
        inference_file=inference_file, 
        bulk_inference=bulk_inference,
        device='cuda:3'
    )
    inference.main()