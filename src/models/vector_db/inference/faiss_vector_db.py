import faiss
import numpy as np
import torch

class FaissVectorDB:
    
    def __init__(self, device: str = 'cpu'):
        self.index = None
        self.paragraph_key_mapping = {}

        if device.startswith('cuda'):
            self.gpu_id = int(device.split(':')[1])
            self.res = faiss.StandardGpuResources()
            self.use_gpu = True
        else:
            self.gpu_id = None
            self.res = None
            self.use_gpu = False
    
    def build_index(self, vectors, unique_keys):
        vectors = vectors.astype('float32')
        vector_dimension = vectors.shape[1]

        if self.index is None:
            if self.use_gpu:
                cpu_index = faiss.IndexFlatL2(vector_dimension)
                self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index)
            else:
                self.index = faiss.IndexFlatL2(vector_dimension)

        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        current_index_size = len(self.paragraph_key_mapping)
        
        assert len(vectors) == len(unique_keys), "Mismatch between number of vectors and unique keys"

        for i, key in enumerate(unique_keys):
            self.paragraph_key_mapping[current_index_size + i] = key
    
    def perform_search(self, query, number_of_results=5, datapoint: dict={}):
        

        query = query.astype('float32')
        faiss.normalize_L2(query)

        distances, ann = self.index.search(query, k=self.index.ntotal)

        paragraph_keys = [self.paragraph_key_mapping[idx] for idx in ann[0]]
        relevant_paragraph_keys = self.obtain_relevant_paras(all_paragraphs_keys=paragraph_keys, number_of_results=number_of_results, datapoint=datapoint)
        return relevant_paragraph_keys
    
    def obtain_relevant_paras(self, all_paragraphs_keys: list, number_of_results: int, datapoint) -> list:
        all_relevant_paragraphs = []
        for para_keys in all_paragraphs_keys:
            if datapoint["link"] in para_keys:
                all_relevant_paragraphs.append(para_keys)
        return all_relevant_paragraphs[:number_of_results]