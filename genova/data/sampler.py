from torch.utils.data import Sampler
from typing import Iterator, List


class GenovaSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, cfg, gpu_capacity, max_node_num=256) -> None:
        self.node_num = max_node_num
        self.data_source = data_source
        self.gpu_capacity = gpu_capacity
        self.cfg = cfg
        self.d_relation = self.cfg['encoder']['d_relation']
        self.num_layers = self.cfg['encoder']['num_layers']
        self.expansion = self.cfg['encoder']['edge_encoder']['expansion_factor']
        self.d_edge = self.cfg['encoder']['edge_encoder']['d_edge']
        self.relation_factor = 6 ## >=6
        self.edge_factor = 1.2 ## >=1.2

        self.relation_gpu = self.relation_factor * self.d_relation * \
                            self.node_num * self.node_num * self.num_layers * 4 / 1024 ** 3
        self.edge_gpu = self.edge_factor * 4 * self.node_num * self.node_num * 2 * \
                        (self.d_relation+self.expansion*self.d_edge) / 1024**3


    def __iter__(self) -> Iterator[List[int]]:
        gpu_used = 0
        batch = []
        counter = 0

        for i, d in enumerate(self.data_source):
            num_all_edges = d[0]['rel_type'].shape[0]
            # if num_all_edges > 200000:
            #     continue
            edge_gpu = self.edge_gpu + self.edge_factor * 4 * num_all_edges * (9 + self.expansion) * self.d_edge / 1024**3

            if gpu_used + self.relation_gpu + edge_gpu > self.gpu_capacity:
                counter += 1
                yield batch
                batch = [i]
                gpu_used = self.relation_gpu + edge_gpu
            else:
                gpu_used += self.relation_gpu
                gpu_used += edge_gpu
                batch.append(i)

        if len(batch) > 0:
            yield batch


    # def __len__(self) -> int:
    #     # Can only be called if self.sampler has __len__ implemented
    #     # We cannot enforce this condition, so we turn off typechecking for the
    #     # implementation below.
    #     # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

