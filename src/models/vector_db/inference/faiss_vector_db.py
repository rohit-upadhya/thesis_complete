import faiss
import numpy as np
class FaissVectorDB:
    
    def build_index(self, vectors):
        print("vectors.shape",vectors.shape)
        vector_dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
    def build_search_vector(self, search_vector):
        print(search_vector.shape)
        self.search_vector = np.array([search_vector])
        faiss.normalize_L2(self.search_vector)
    
    def perform_search(self, k):
        distances, ann = self.index.search(self.search_vector, k=k)
        return distances, ann
    
    def main(self, paragraphs, query, number_of_relevant_paragraphs):
        self.build_index(np.array(paragraphs))
        self.build_search_vector(query)
        distances, ann = self.perform_search(number_of_relevant_paragraphs)
        print (distances, ann)
        pass