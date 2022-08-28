import numpy as np
from torch.utils.data import Sampler

class PERSampler(Sampler):
    def __init__(self, dataset, train_batch_size, inference_batch_size, gamma=0.5):
        self.dataset = dataset
        assert train_batch_size>inference_batch_size
        self.inference_batch_size = inference_batch_size
        self.train_batch_size = train_batch_size
        self.sample_idx_number = self.train_batch_size-self.inference_batch_size
        self.p = (1/np.arange(1,self.dataset.capacity+1))**gamma
        self.p = self.p/self.p.sum()

    def __iter__(self):
        return self
    
    def __next__(self):
        self.dataset.memory_pool_update()
        persample_idx = np.random.choice(self.dataset.capacity, size=self.sample_idx_number, p=self.p, replace=False)
        new_idx = np.arange(self.dataset.capacity,len(self.dataset.memory))
        return np.concatenate([new_idx,persample_idx])