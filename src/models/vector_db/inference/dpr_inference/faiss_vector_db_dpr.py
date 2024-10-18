import faiss # type: ignore
import numpy as np # type: ignore

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

        assert len(vectors) == len(unique_keys), "Mismatch between number of vectors and unique keys"

        for i, key in enumerate(unique_keys):
            self.paragraph_key_mapping[ids[i]] = key
    
    def perform_search(self, query, datapoint: dict={}):
        

        query = query.astype('float32')
        faiss.normalize_L2(query)

        distances, ann = self.index.search(query, k=self.index.ntotal) # type: ignore
        paragraph_keys = [self.paragraph_key_mapping[idx] for idx in ann[0]]
        relevant_paragraph_keys = self.obtain_relevant_paras(all_paragraphs_keys=paragraph_keys, datapoint=datapoint)
        return relevant_paragraph_keys
        
    def obtain_relevant_paras(self, all_paragraphs_keys: list, datapoint) -> list:
        all_relevant_paragraphs = []
        for para_keys in all_paragraphs_keys:
            if datapoint["link"] in para_keys:
                all_relevant_paragraphs.append(para_keys)
        return all_relevant_paragraphs