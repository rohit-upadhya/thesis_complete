import faiss  # type: ignore
import numpy as np  # type: ignore

class FaissVectorDB:
    
    def __init__(self, device: str = 'cpu'):
        self.index = None
        self.paragraph_key_mapping = {}
        self.use_gpu = False

        if device.startswith('cuda'):
            self.gpu_id = int(device.split(':')[1])
            self.res = faiss.StandardGpuResources()
            self.use_gpu = True
        else:
            self.gpu_id = None
            self.res = None
    
    def build_index(self, vectors):
        vectors = vectors.astype('float32')
        vector_dimension = vectors.shape[1]

        faiss.normalize_L2(vectors)

        if self.index is None:
            if self.use_gpu:
                cpu_index = faiss.IndexFlatIP(vector_dimension)
                cpu_index = faiss.IndexIDMap(cpu_index)
                self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index)
            else:
                cpu_index = faiss.IndexFlatIP(vector_dimension)
                self.index = faiss.IndexIDMap(cpu_index)

        if self.index.ntotal == 0:
            start_id = 0
        else:
            start_id = int(self.index.ntotal)

        ids = np.arange(start_id, start_id + len(vectors)).astype('int64')
        self.index.add_with_ids(vectors, ids)
    
    def perform_search(self, query, datapoint: dict={}):
        query = query.astype('float32')
        faiss.normalize_L2(query)

        distances, ann = self.index.search(query, k=self.index.ntotal)  # type: ignore
        relevant_paragraph_keys = [idx for idx in ann[0]]  # Now simply return indices
        return relevant_paragraph_keys
