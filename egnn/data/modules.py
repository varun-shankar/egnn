import torch
from typing import Optional
import random
from torch_cluster import knn_graph, radius_graph
import glob, re, os
import torch.nn.functional as F
from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch
from torch_geometric.utils import degree
from e3nn.math import soft_one_hot_linspace
from math import log
from e3nn import o3, io
from torch_geometric.loader import DataLoader, NeighborSampler, RandomNodeSampler
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
import pytorch_lightning as pl


class Data(pygData):
    # def __cat_dim__(self, key, value, *args, **kwargs):
    #     if key == 'y':
    #         return 1
    #     else:
    #         return super().__cat_dim__(key, value, *args, **kwargs)

    def rotate(self, rot):
        irreps_in = o3.Irreps(self.irreps_io[0][0]).simplify()
        irreps_out = o3.Irreps(self.irreps_io[0][1]).simplify()
        irreps_fn = o3.Irreps(self.irreps_io[0][2]).simplify()
        D_in = irreps_in.D_from_matrix(rot).type_as(self.x)
        D_out = irreps_out.D_from_matrix(rot).type_as(self.x)
        D_fn = irreps_fn.D_from_matrix(rot).type_as(self.x)
        self.x = self.x @ D_in.T
        self.pos = self.pos @ rot.type_as(self.x).T
        self.y = self.y @ D_out.T
        self.fn = self.fn @ D_fn.T
        return self, D_out

    def embed(self, rc, num_fes=16, irreps_sh=o3.Irreps.spherical_harmonics(lmax=2)):
        edge_src, edge_dst = self.edge_index
        deg = degree(edge_dst, self.num_nodes).type_as(self.hn)
        # print(deg.min())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm = deg_inv_sqrt[edge_src] * deg_inv_sqrt[edge_dst]
        self.norm = self.norm * self.edge_norm if 'edge_norm' in self else self.norm

        self.edge_vec = self.pos[edge_dst] - self.pos[edge_src]
        # rc = float(self.rc.max()) if torch.is_tensor(self.rc) else self.rc
        self.fes = soft_one_hot_linspace(self.edge_vec.norm(dim=1), 0.0, rc, num_fes, 
            basis='smooth_finite', cutoff=False).mul(num_fes**0.5)
        self.fe = o3.spherical_harmonics(irreps_sh, self.edge_vec, normalize=True, normalization='component')

    def resample_edges(self, rkm, training=True):
        with torch.no_grad():
            r, k, m = rkm
            if isinstance(m, list):
                m = m[0] if training else m[1]
            if k != 0:
                self.edge_index = knn_graph(self.pos, batch=self.batch, k=k)
            else:
                self.edge_index = radius_graph(
                    self.pos, batch=self.batch, r=r, max_num_neighbors=m)
        avg_nbr = self.edge_index.shape[1]/self.pos.shape[0]
        if avg_nbr < 5: print(f'Warning: avg neighbors less than 5 ({avg_nbr:.2f})')
        self.embed(r)
        return self

    def subsample(self):
        num_nodes = torch.randint(5000,8000,(1,))
        with torch.no_grad():
            dlist = self.to_data_list()
            for i in range(self.num_graphs):
                subset = torch.randperm(dlist[i].num_nodes)[:num_nodes]
                dlist[i] = dlist[i].subgraph(subset)
                # dlist[i].edge_index = subgraph(subset, dlist[i].edge_index, relabel_nodes=True)
                # dlist[i].pos = dlist[i].pos[subset]
                # dlist[i].x = dlist[i].x[subset]
                # dlist[i].y = dlist[i].y[subset]
                # dlist[i].fn = dlist[i].fn[subset]
        return pygBatch.from_data_list(dlist)

def check_sampled_data(dataset):
    nn = 0; ne = 0
    for i in range(len(dataset)):
        nn += dataset[i].num_nodes
        ne += dataset[i].edge_index.size(1)
    print(f'Sampled data: {nn/len(dataset):.2f} nodes, {ne/len(dataset):.2f} edges')

class RWSampled_Dataset(RWSampler):
    def __init__(self, data, **kwargs):
        self.finished_loading = False
        super().__init__(data, **kwargs)
        self.finished_loading = True
        self.g = torch.Generator(); self.g.seed()
        _ = check_sampled_data(self) if int(os.environ.get('LOCAL_RANK', 0)) == 0 else 0
    def __getitem__(self, idx):
        if self.finished_loading:
            sample_ids = (self.data.batch==idx).nonzero().flatten()
            start = sample_ids[torch.randint(0, sample_ids.size(0), 
                (self.__batch_size__,), generator=self.g, dtype=torch.long)]
            n_id = self.adj.random_walk(start.flatten(), self.walk_length).view(-1)
            adj, _ = self.adj.saint_subgraph(n_id)
            data = RWSampler.__collate__(self, [(n_id, adj)])
            data.irreps_io = data.irreps_io[0]
            return data
        else:
            return RWSampler.__getitem__(self, idx)

class Cluster_Dataset(NeighborSampler):
    def __init__(self, data, batch_size=1, num_steps=1, **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.N = data.num_nodes
        self.E = data.edge_index.size(1)
        super().__init__(data.edge_index, **kwargs)
        self.sample_coverage = 0
        self.g = torch.Generator(); self.g.seed()
        _ = check_sampled_data(self) if int(os.environ.get('LOCAL_RANK', 0)) == 0 else 0
    def __len__(self):
        return self.num_steps
    def __getitem__(self, idx):
        sample_ids = (self.data.batch==idx).nonzero().flatten()
        n_id = self.sample(sample_ids[torch.randint(0, sample_ids.size(0), 
            (self.batch_size,), generator=self.g, dtype=torch.long)])[1]
        adj, _ = self.adj_t.saint_subgraph(n_id)
        data = RWSampler.__collate__(self, [(n_id, adj)])
        data.irreps_io = data.irreps_io[0]
        return data.subgraph(
            torch.randperm(
                data.num_nodes)[:torch.randint(int(0.5*data.num_nodes),data.num_nodes,(1,))]
        )

