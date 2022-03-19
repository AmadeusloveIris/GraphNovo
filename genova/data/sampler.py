import random
from torch.utils.data import Sampler
from typing import Iterator, List


class GenovaSampler(Sampler[List[int]]):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, cfg, gpu_capacity, scale_factor=1.37, error_tol=0.2) -> None:
        self.data_source = data_source
        self.bins = self.generate_bins(data_source)
        self.cfg = cfg
        self.gpu_capacity = gpu_capacity / scale_factor
        self.error_tol = error_tol
        self.hidden_size = self.cfg['hidden_size']
        self.d_relation = self.cfg['encoder']['d_relation']
        self.num_layers = self.cfg['encoder']['num_layers']
        self.expansion = self.cfg['encoder']['edge_encoder']['expansion_factor']
        self.d_edge = self.cfg['encoder']['edge_encoder']['d_edge']

        self.relation_gpu1 = 4 * 3 * self.d_relation * self.num_layers / 1024 ** 3 ## node_num**2
        self.relation_gpu2 = 4 * (3 * self.d_relation + 11 * self.hidden_size) * \
                             self.num_layers / 1024 ** 3  ## node_num
        self.edge_gpu1 = 4 * 2 * (self.d_relation+self.expansion*self.d_edge) / 1024**3 ## node_num**2
        self.edge_gpu2 = 4 * (9 + self.expansion) * self.d_edge / 1024**3 ## num_all_edges


    def __iter__(self) -> Iterator[List[int]]:
        edge_gpu_used1 = 0
        edge_gpu_used2 = 0
        relation_gpu_used1 = 0
        relation_gpu_used2 = 0
        batch = []
        max_node = 1

        counter = 0

        bin_tracker = {0: 0, 1: 0, 2: 0}
        bin_len = {0: len(self.bins[0]), 1: len(self.bins[1]), 2: len(self.bins[2])}
        bin_len_left = {l: bin_len[l] - bin_tracker[l] for l in [0, 1, 2]}

        while bin_len_left[0] or bin_len_left[1] or bin_len_left[2]:
            # print(bin_len_left)
            which_bin = random.choices([0, 1, 2], weights=[bin_len_left[l] for l in bin_len_left])[0]
            bin_index = bin_tracker[which_bin]

            while bin_index < bin_len[which_bin]:
                i = self.bins[which_bin][bin_index] ## index in data_source
                d = self.data_source[i]
                num_all_edges = d[0]['rel_type'].shape[0]
                node_num = d[0]['node_feat'].shape[0]

                # if num_all_edges > 200000:
                #     continue

                gpu_used_scale = 1.0 * max(max_node, node_num) / max_node
                max_node = max(max_node, node_num)

                relation_gpu1 = self.relation_gpu1 * max_node * max_node
                relation_gpu2 = self.relation_gpu2 * max_node
                edge_gpu1 = self.edge_gpu1 * max_node * max_node
                edge_gpu2 = self.edge_gpu2 * num_all_edges

                relation_gpu_used1 = relation_gpu_used1 * gpu_used_scale**2
                relation_gpu_used2 = relation_gpu_used2 * gpu_used_scale
                edge_gpu_used1 = edge_gpu_used1 * gpu_used_scale**2

                if relation_gpu_used1 + relation_gpu_used2 + edge_gpu_used1 + edge_gpu_used2 + \
                        relation_gpu1 + relation_gpu2 + edge_gpu1 + edge_gpu2 > self.gpu_capacity - self.error_tol:
                    counter += 1
                    if counter % 20 == 0:
                        print('which bin: ', which_bin, batch)
                    yield batch
                    # batch = [i]
                    # relation_gpu_used1 = self.relation_gpu1 * node_num * node_num
                    # relation_gpu_used2 = self.relation_gpu2 * node_num
                    # edge_gpu_used1 = self.edge_gpu1 * node_num * node_num
                    # edge_gpu_used2 = edge_gpu2
                    # max_node = node_num

                    edge_gpu_used1 = 0
                    edge_gpu_used2 = 0
                    relation_gpu_used1 = 0
                    relation_gpu_used2 = 0
                    batch = []
                    max_node = 1

                    bin_tracker[which_bin] = bin_index
                    bin_len_left = {l: bin_len[l] - bin_tracker[l] for l in [0, 1, 2]}
                    break
                else:
                    edge_gpu_used1 += edge_gpu1
                    edge_gpu_used2 += edge_gpu2
                    relation_gpu_used1 += relation_gpu1
                    relation_gpu_used2 += relation_gpu2
                    batch.append(i)

                bin_index += 1

            if len(batch) > 0:
                counter += 1
                print('which bin: ', which_bin, batch)
                yield batch

                edge_gpu_used1 = 0
                edge_gpu_used2 = 0
                relation_gpu_used1 = 0
                relation_gpu_used2 = 0
                batch = []
                max_node = 1

                bin_tracker[which_bin] = bin_index
                bin_len_left = {l: bin_len[l] - bin_tracker[l] for l in [0, 1, 2]}


    # def __len__(self) -> int:
    #     # Can only be called if self.sampler has __len__ implemented
    #     # We cannot enforce this condition, so we turn off typechecking for the
    #     # implementation below.
    #     # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

    def generate_bins(self, data_source):
        bins = [[] for _ in range(3)]
        for i, d in enumerate(data_source):
            if d[0]['node_feat'].shape[0] < 128:
                bins[0].append(i)
            elif d[0]['node_feat'].shape[0] < 256:
                bins[1].append(i)
            elif d[0]['node_feat'].shape[0] < 512:
                bins[2].append(i)

        # for bin in bins:
        #     random.shuffle(bin)
        return bins
