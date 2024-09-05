from torch.utils.data.sampler import BatchSampler
import numpy as np

class BatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

        self.query_ids = []
        self.positive_indices = []
        self.negative_indices = []
        
        for idx, element in enumerate(dataset):
            self.query_ids.append(element["qid"])
            if element["labels"] == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)

        self.unique_queries = list(set(self.query_ids))
    def __iter__(self):
        np.random.shuffle(self.unique_queries)
        
        for query in self.unique_queries:
            pos_samples = [i for i in self.positive_indices if self.dataset[i]['qid']==query]
            neg_samples = [i for i in self.negative_indices if self.dataset[i]['qid']==query]
            
            if not pos_samples or not neg_samples:
                continue
            
            for pos in pos_samples:
                if len(neg_samples) >= self.batch_size - 1:
                    batch = [pos] + neg_samples[:self.batch_size - 1]
                else:
                    batch = [pos] + neg_samples
                    batch += neg_samples[:self.batch_size-1-len(batch)]
                batch = batch[:self.batch_size]
                yield batch
    
    def __len__(self):
        return len(self.unique_queries)