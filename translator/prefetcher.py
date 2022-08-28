import torch
import collections
from typing import Union
from torch._six import string_classes

class DataPrefetcher:
    def __init__(self, loader, device: Union[int, torch.device], valid: bool):
        """Data Prefetcher for prefetch data to GPU.

        Args:
            loader (torch.DataLoader)
            device (torch.device): Device where you want to move.

        Warning:
            You have to pin memory before use this.
        """
        self.valid = valid
        self.device = device
        self.loader_ori = loader
        self.stream = torch.cuda.Stream()

    def preload(self):
        self.batch = next(self.loader)
        if self.valid: self.batch = self.to_cuda(self.batch)
        else:
            self.idx = self.batch[-1]
            self.batch = self.to_cuda(self.batch[:-1])
    
    def __iter__(self):
        self.loader = iter(self.loader_ori)
        self.end = False
        self.preload()
        return self

    def __next__(self):
        # You need to wait until last batch of data
        # move to device
        if self.end: raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if not self.valid: batch.append(self.idx)
        try: self.preload()
        except StopIteration: self.end = True
        return batch
    
    def to_cuda(self, data):
        with torch.cuda.stream(self.stream):
            if isinstance(data, torch.Tensor):
                return data.to(device=self.device, non_blocking=True)
            elif isinstance(data, string_classes):
                return data
            elif isinstance(data, collections.abc.Mapping):
                try:
                    return type(data)({k: self.to_cuda(sample) for k, sample in data.items()})  # type: ignore[call-arg]
                except TypeError:
                    # The mapping type may not support `__init__(iterable)`.
                    return {k: self.to_cuda(sample) for k, sample in data.items()}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return type(data)(*(self.to_cuda(sample) for sample in data))
            elif isinstance(data, tuple):
                return [self.to_cuda(sample) for sample in data]  # Backwards compatibility.
            elif isinstance(data, collections.abc.Sequence):
                try:
                    return type(data)([self.to_cuda(sample) for sample in data])  # type: ignore[call-arg]
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.to_cuda(sample) for sample in data]
            elif hasattr(data, "to"):
                return data.to(device=self.device, non_blocking=True)
            else:
                return data

class DataTransfer(object):
    def __init__(self, loader, device: Union[int, torch.device]):
        self.device = device
        self.loader_ori = loader

    def __iter__(self):
        self.loader = iter(self.loader_ori)
        return self
    
    def __next__(self):
        # You need to wait until last batch of data
        # move to device
        idx, model_input, label = next(self.loader)
        model_input, label = self.to_cuda(model_input), self.to_cuda(label)
        return idx, model_input, label

    def to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        elif isinstance(data, string_classes):
            return data
        elif isinstance(data, collections.abc.Mapping):
            try:
                return type(data)({k: self.to_cuda(sample) for k, sample in data.items()})  # type: ignore[call-arg]
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {k: self.to_cuda(sample) for k, sample in data.items()}
        elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
            return type(data)(*(self.to_cuda(sample) for sample in data))
        elif isinstance(data, tuple):
            return [self.to_cuda(sample) for sample in data]  # Backwards compatibility.
        elif isinstance(data, collections.abc.Sequence):
            try:
                return type(data)([self.to_cuda(sample) for sample in data])  # type: ignore[call-arg]
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [self.to_cuda(sample) for sample in data]
        elif hasattr(data, "to"):
            return data.to(device=self.device)
        else:
            return data