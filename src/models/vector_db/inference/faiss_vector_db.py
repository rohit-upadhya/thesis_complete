import faiss
import numpy as np

class FaissVectorDB:
    
    def build_index(self, vectors):
        vectors = vectors.astype('float32')
        
        vector_dimension = vectors.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        
    def build_search_vector(self, search_vector):
        if len(search_vector.shape) == 1:
            self.search_vector = np.expand_dims(search_vector, axis=0).astype('float32')
        else:
            self.search_vector = search_vector.astype('float32')
        faiss.normalize_L2(self.search_vector)
    
    def perform_search(self, k):
        distances, ann = self.index.search(self.search_vector, k=k)
        return distances, ann
    
    def main(self, paragraphs, query, number_of_relevant_paragraphs=5):
        self.build_index(np.array(paragraphs))
        self.build_search_vector(np.array(query))
        distances, ann = self.perform_search(number_of_relevant_paragraphs)
        return distances[0], ann[0]
