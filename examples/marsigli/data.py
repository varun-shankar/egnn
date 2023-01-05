import torch
import numpy as np
from typing import Optional
import random, os
from torch_cluster import radius_graph
import torch.nn.functional as F
from torch_geometric.data import Batch as pygBatch
from math import log
from e3nn import o3, io
from egnn.data.modules import *
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl


def load_case(dir, Re, ts, data_fields):
    lx = 8
    ly = 1
    nx = 512
    ny = int(nx/8)
    dx = lx/nx
    dy = ly/ny 
    x = np.linspace(0.0,lx,nx+1)
    y = np.linspace(0.0,ly,ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X, Y = X[::2,::2], Y[::2,::2]
    p = torch.stack([torch.tensor(X.flatten()),torch.tensor(Y.flatten()),torch.tensor(0*X.flatten())],dim=-1)

    b = X*0; b[:,[0,-1]]=1; b[[0,-1],:]=1
    b = torch.tensor(b.flatten()).long()

    v = []
    for i in ts:
        out = np.load(dir+'Re_'+str(Re)+'/data_512_64/data_'+str(i.item())+'.npz')
        vt = torch.stack([torch.tensor(out[df][::2,::2].flatten()) for df in ['w','s','t']],dim=-1)
        if data_fields == 'uvt':
            uvw = []
            for u in velocity(nx,ny,dx,dy,out['s']):
                tmp = np.zeros([nx+1,ny+1])
                tmp[1:-1,1:-1] = u
                uvw.append(torch.tensor(tmp[::2,::2].flatten()))
            uvw = torch.stack([uvw[0],uvw[1],0*uvw[0]],dim=-1)
            vt = torch.cat([uvw,vt[:,-1:]],dim=-1)
        v.append(vt)
    
    v = torch.stack(v)
    return p.float(), b, v.float()

def velocity(nx,ny,dx,dy,s):
    u =  np.zeros([nx-1,ny-1])
    v =  np.zeros([nx-1,ny-1])
    # u = ds/dy
    u = (s[1:nx,2:ny+1] - s[1:nx,0:ny-1])/(2*dy)
    # v = -ds/dx
    u = -(s[2:nx+1,1:ny] - s[0:nx-1,1:ny])/(2*dx)
    return u,v


class DataModule(pl.LightningDataModule):
    def __init__(self, tspan, rkm,
                       dir='/home/vshanka2/data/marsigli/Results/',
                       num_nodes=-1, sample_graph=None,
                       rollout=1,
                       data_fields='wst', irreps_fn='3x0e+1o',
                       train_split=0.9, random_split=True,
                       train_cases=[700,900,1100,1300],
                       test_case=1000, test_rollout=None,
                       shuffle=True, batch_size=1, **kwargs):
        super().__init__()
        self.dir = dir
        self.tspan = tspan
        self.rkm = rkm
        self.num_nodes = num_nodes
        self.sample_graph = sample_graph
        self.rollout = rollout
        self.data_fields = data_fields
        self.train_split = train_split
        self.random_split = random_split
        self.train_cases = train_cases
        self.test_case = test_case
        self.test_rollout = tspan[1]-tspan[0] if test_rollout is None else test_rollout
        self.shuffle = shuffle
        self.batch_size = batch_size if isinstance(batch_size,int) else 1
        self.irreps_data = ['1o+0e','1o+0e'] if self.data_fields=='uvt' else ['3x0e','3x0e']
        self.irreps_io = self.irreps_data; self.irreps_io.append(irreps_fn)
        self.model_args = {'irreps_io': self.irreps_io}
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.ts = 80*(torch.arange(self.tspan[0],self.tspan[1]+1))*2
            dt = 5e-4
            n = o3.Norm(self.irreps_data[0])
            mult = o3.ElementwiseTensorProduct(n.irreps_out, self.irreps_data[0])

            ### Read data ###
            dataset = []
            for case in self.train_cases:
                p, b, v = load_case(self.dir, case, self.ts, self.data_fields)
                if self.num_nodes != -1:
                    idx = torch.randperm(p.shape[0])[:self.num_nodes]
                    p = p[idx,:]; b = b[idx]; v = v[:,idx,:]
                
                pos = p
                v = mult(1/n(v).amax(dim=(0,1),keepdim=True),v)
                b1hot = F.one_hot(b).float()
                fn = torch.cat([b1hot,
                    log(case)*torch.ones(pos.shape[0],1), #pos-pos.mean(0, keepdim=True),
                    torch.tensor([0,-1,0]).repeat(pos.shape[0],1)],dim=-1)

                # Generate graph
                edge_index = radius_graph(pos, r=self.rkm[0][0], max_num_neighbors=25)
                # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                #     print(f'Avg neighbors: {edge_index.shape[1]/pos.shape[0]:.2f}')

                set = [Data(x=v[i,:,:], fn=fn,
                            y=v[i+1:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(dt*self.ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rkm=self.rkm
                            ) for i in range(len(self.ts)-self.rollout)]

                dataset.extend(set)

            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            if self.sample_graph != None:
                self.train_data = pygBatch.from_data_list(self.train_data)
                if self.sample_graph == 'random walk':
                    self.train_data = RWSampled_Dataset(self.train_data, 
                        batch_size=getattr(self,'batch_size',1000), num_steps=self.train_data.num_graphs,
                        walk_length=getattr(self,'hops',getattr(self,'latent_layers',10)), 
                        sample_coverage=getattr(self,'sample_coverage',0),
                        save_dir='.')
                elif self.sample_graph == 'cluster':
                    self.train_data = Cluster_Dataset(self.train_data, 
                        batch_size=getattr(self,'batch_size',1), num_steps=self.train_data.num_graphs,
                        sizes=-1*torch.ones(getattr(self,'hops',getattr(self,'latent_layers',10))))
                else:
                    print('Unknown sampling method')

            # Test
            p, b, v = load_case(self.dir, self.test_case, self.ts, self.data_fields)
            if self.num_nodes != -1:
                idx = torch.randperm(p.shape[0])[:self.num_nodes]
                p = p[idx,:]; b = b[idx]; v = v[:,idx,:]

            pos = p
            v = mult(1/n(v).amax(dim=(0,1),keepdim=True),v)
            b1hot = F.one_hot(b).float()
            fn = torch.cat([b1hot,
                log(self.test_case)*torch.ones(pos.shape[0],1), #pos-pos.mean(0, keepdim=True),
                torch.tensor([0,-1,0]).repeat(pos.shape[0],1)],dim=-1)

            # Generate graph
            edge_index = radius_graph(pos, r=self.rkm[0][0], max_num_neighbors=25)
            # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            #     print(f'Avg neighbors: {edge_index.shape[1]/pos.shape[0]:.2f}')

            testset = [Data(x=v[i,:,:], fn=fn,
                            y=v[i+1:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(dt*self.ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rkm=self.rkm
                            ) for i in range(len(self.ts)-self.rollout)]
            testset_rollout = [Data(x=v[i,:,:], fn=fn,
                                y=v[i+1:i+1+self.test_rollout,:,:].transpose(0,1), 
                                irreps_io=self.irreps_io,
                                ts=(dt*self.ts[i:i+1+self.test_rollout].unsqueeze(0)), 
                                pos=pos, edge_index=edge_index, rkm=self.rkm
                                ) for i in range(len(self.ts)-self.test_rollout)]
            self.test_data = testset
            self.test_data_rollout = testset_rollout

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return [DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4),
                DataLoader(self.test_data_rollout, num_workers=4)]