from torch.utils.data.sampler import BatchSampler
import numpy as np

class BatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, batch_counts) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_counts = batch_counts
        self.query_pos_indices = {}
        self.query_neg_indices = {}
        for idx, element in enumerate(dataset):
            query_id = element["qid"]
            if query_id not in self.query_pos_indices:
                self.query_pos_indices[query_id] = []
                self.query_neg_indices[query_id] = []

            if element["labels"] == 1:
                self.query_pos_indices[query_id].append(idx)
            else:
                self.query_neg_indices[query_id].append(idx)

        self.unique_queries = list(self.query_pos_indices.keys())
        
    # def __iter__(self):
    #     np.random.shuffle(self.unique_queries)
        
    #     for query in self.unique_queries:
    #         pos_samples = self.query_pos_indices.get(query, [])
    #         neg_samples = self.query_neg_indices.get(query, [])
            
    #         if not pos_samples or not neg_samples:
    #             continue
            
    #         for pos in pos_samples:
    #             if len(neg_samples) >= self.batch_size - 1:
    #                 batch = [pos] + neg_samples[:self.batch_size - 1]
    #             else:
    #                 batch = [pos] + neg_samples
    #                 batch += neg_samples[:self.batch_size-1-len(batch)]
    #             if len(batch)>self.batch_size:
    #                 batch = batch[:self.batch_size]
    #             print("insode batching ",len(batch))
    #             yield batch
    def __iter__(self):
        np.random.shuffle(self.unique_queries)
        
        for query in self.unique_queries:
            pos_samples = self.query_pos_indices.get(query, [])
            neg_samples = self.query_neg_indices.get(query, [])
            np.random.shuffle(neg_samples)
            
            if not pos_samples or not neg_samples:
                continue
            
            for pos in pos_samples:
                batch = [pos]
                
                if len(neg_samples) >= self.batch_size - 1:
                    batch += neg_samples[:self.batch_size - 1]
                else:
                    batch += neg_samples
                    while len(batch) < self.batch_size:
                        batch.append(np.random.choice(neg_samples))
                
                if len(batch) > self.batch_size:
                    batch = batch[:self.batch_size]
                
                yield batch
    
    def __len__(self):
        return self.batch_counts
