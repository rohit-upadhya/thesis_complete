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

    # def build_search_vector(self, search_vector):
    #     if len(search_vector.shape) == 1:
    #         self.search_vector = np.expand_dims(search_vector, axis=0).astype('float32')
    #     else:
    #         self.search_vector = search_vector.astype('float32')
    #     faiss.normalize_L2(self.search_vector)
    
    # def perform_search(self, k):
    #     distances, ann = self.index.search(self.search_vector, k=k)
    #     return distances, ann
    
    def perform_search_with_indices(self, query, subset_indices, number_of_relevant_paragraphs=5):
        # Extract the relevant vectors on CPU and perform the search there
        subset_vectors = np.take(self.index.reconstruct_n(0, self.index.ntotal), subset_indices, axis=0)
        index_subset = faiss.IndexFlatL2(subset_vectors.shape[1])  # Create CPU FAISS index for subset search
        index_subset.add(subset_vectors)

        # Perform the search on the subset
        distances, ann = index_subset.search(query, number_of_relevant_paragraphs)

        # Map the results back to the original FAISS index's keys
        relevant_paragraph_keys = [self.paragraph_key_mapping[subset_indices[idx]] for idx in ann[0]]
        return distances[0], relevant_paragraph_keys
    
    # def main(self, paragraphs, query, unique_keys, number_of_relevant_paragraphs=5):
    #     self.build_index(np.array(paragraphs), unique_keys)
    #     self.build_search_vector(np.array(query))
    #     distances, ann = self.perform_search(number_of_relevant_paragraphs)
    #     print(len(self.paragraph_key_mapping))
    #     relevant_paragraph_keys = [self.paragraph_key_mapping[idx] for idx in ann[0]]
    #     return distances[0], relevant_paragraph_keys
