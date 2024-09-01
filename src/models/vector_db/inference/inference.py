from typing import Text, List, Dict, Any
import torch

from src.models.vector_db.commons.input_loader import InputLoader
from src.models.vector_db.inference.encoder import Encoder
from src.models.vector_db.inference.faiss_vector_db import FaissVectorDB

class Inference:
    def __init__(self, inference_file:Text=None, inference_datapoints:List[Dict[str,Any]]=None, bulk_inference:bool=False):
        inference_datapoints = inference_datapoints
        input_loader = InputLoader()
        if bulk_inference:
            inference_datapoints = input_loader.load_data(inference_file)
        self._format_input(inference_datapoints)
    
    def _format_input(self, inference_datapoints):
        self.data_points = []
        for data in inference_datapoints:
            self.data_points.append(
                {
                    "query": " ".join(data["query"]),
                    "all_paragraphs": [" ".join(paras) for paras in data["all_paragraphs"]]
                }
            )
    def _encode_queries_paragraphs(self, query: Text, all_paragraphs: List[Text], query_encoder: Encoder, parar_encoder: Encoder):
        query_encodings = query_encoder.encode([query])
        all_paragraph_encodings = parar_encoder.encode(all_paragraphs)
        return query_encodings, all_paragraph_encodings
    
    def main(self):
        encoder = Encoder()
        results = []
        for points in self.data_points:
            query_encodings, all_paragraph_encodings = self._encode_queries_paragraphs(query=points["query"], all_paragraphs=points["all_paragraphs"], query_encoder=encoder, parar_encoder=encoder)
            faiss = FaissVectorDB()
            faiss.main(paragraphs=all_paragraph_encodings, query=query_encodings, number_of_relevant_paragraphs=5)

if __name__=="__main__":
    inference_file = "/home/upadro/code/thesis/src/models/single_datapoints/common/test.json"
    bulk_inference = True
    inference = Inference(inference_file=inference_file, bulk_inference=bulk_inference)
    inference.main()
    pass