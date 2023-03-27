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
import fluidfoam as ff
import pyvista as pv
import sys

class Suppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if type is not None:
            raise
    def write(self, x): pass

def read_field_snap(dir,t,f,b,numpts):
    tstr = f'{t:g}'
    if f=='streamFunction':
        if b is None:
            mesh = pv.read(dir+'/VTK/cylinder2D_base_'+tstr+'/internal.vtu')
            v = torch.tensor(mesh.point_data_to_cell_data()['streamFunction'])
        else:
            mesh = pv.read(dir+'/VTK/cylinder2D_base_'+tstr+'/boundary/'+b+'.vtp')
            v = torch.tensor(mesh.point_data_to_cell_data()['streamFunction'])
    else:
        v = torch.tensor(ff.readfield(dir,tstr,f,boundary=b)).float()
    v = v.t() if v.dim() == 2 else v.unsqueeze(-1)
    v = v.repeat(numpts,1) if v.shape[0] == 1 else v
    if f=='vorticity':
        v = v[:,~torch.isclose(v.abs().sum(dim=0),torch.tensor(0.))]
    return v

def read_mesh_and_field(dir,b,df,ts):
    fields = ['U','p'] if df=='uvp' else ['vorticity','streamFunction','p']
    p = torch.tensor(np.array(ff.readmesh(dir,boundary=b))).t().float()
    numpts = p.shape[0]
    v = torch.stack([torch.cat([read_field_snap(dir,t,f,b,numpts) for f in fields],dim=-1) for t in ts])
    return p, v, numpts

def load_OF(dir,fields=[],ts=[0.],bounds=['internal'],verbose=False):
    bounds.remove('internal') if 'internal' in bounds else False
    bounds.insert(0,'internal')
    p = []
    b_ind = []
    v = []

    i = 0
    for b in bounds:
        b = None if b == 'internal' else b
        if verbose:
            pb, vb, numb = read_mesh_and_field(dir,b,fields,ts)
        else:
            with Suppressor():
                pb, vb, numb = read_mesh_and_field(dir,b,fields,ts)
        p.append(pb)
        b_ind.append((i*torch.ones(numb)).long())
        v.append(vb)
        i+=1

    return p, b_ind, v


class DataModule(pl.LightningDataModule):
    def __init__(self, ts, rkm,
                       dt=0.02,
                       dir='/home/vshanka2/data/cylinder2D_base/',
                       zones=['internal','cylinder','inlet','outlet','top','bottom'],
                       num_nodes=-1, sample_graph=None,
                       rollout=1,
                       data_fields='uvp', irreps_fn='6x0e+1o',
                       train_split=0.9, random_split=True,
                       test_ts=[], test_rollout=None,
                       shuffle=True, batch_size=1, **kwargs):
        super().__init__()
        self.dir = dir
        self.ts = ts
        self.rkm = rkm
        self.dt = dt
        self.zones = zones
        self.num_nodes = num_nodes
        self.sample_graph = sample_graph
        self.rollout = rollout
        self.data_fields = data_fields
        self.train_split = train_split
        self.random_split = random_split
        self.test_ts = test_ts
        self.test_rollout = int((test_ts[1]-test_ts[0])/dt) if test_rollout is None else test_rollout
        self.shuffle = shuffle
        self.batch_size = batch_size if isinstance(batch_size,int) else 1
        self.irreps_data = ['1o+0e','1o+0e'] if self.data_fields=='uvp' else ['3x0e','3x0e']
        self.irreps_io = self.irreps_data; self.irreps_io.append(irreps_fn)
        self.model_args = {'irreps_io': self.irreps_io}
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.ts = self.ts if torch.is_tensor(self.ts) else torch.arange(self.ts[0],
                self.ts[1]+1e-8,step=self.dt)
            self.test_ts = self.test_ts if torch.is_tensor(self.test_ts) else torch.arange(self.test_ts[0],
                self.test_ts[1]+1e-8,step=self.dt)
            all_ts = torch.cat([self.ts,self.test_ts],dim=0)
            n = o3.Norm(self.irreps_data[0])
            mult = o3.ElementwiseTensorProduct(n.irreps_out, self.irreps_data[0])

            ### Read data ###
            p, b, v = load_OF(self.dir, self.data_fields, all_ts, self.zones)
            
            # Sample internal nodes
            if self.num_nodes != -1:
                n_bounds = sum([len(b[i+1]) for i in range(len(b)-1)])
                # torch.manual_seed(42)
                idx = torch.randperm(p[0].shape[0])[:self.num_nodes-n_bounds]
                p[0] = p[0][idx,:]; b[0] = b[0][idx]; v[0] = v[0][:,idx,:]

            pos = torch.cat(p,dim=0)
            b1hot = F.one_hot(torch.cat(b,dim=0)).float()
            fn = torch.cat([b1hot,
                           torch.tensor([1,0,0]).repeat(pos.shape[0],1)],dim=-1)
            v = torch.cat(v,dim=1)

            # Normalization
            # um = v[:,:,1:].norm(dim=-1).mean()
            # v[:,:,:1] /= um**2
            # v[:,:,1:] /= um
            v = mult(1/n(v).amax(dim=(0,1),keepdim=True),v)

            # Generate graph
            edge_index = radius_graph(pos, r=self.rkm[0][0], max_num_neighbors=25)

            ### Generate dataset ###
            dataset = [Data(x=v[i,:,:], fn=fn,
                            y=v[i:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(all_ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rkm=self.rkm
                            ) for i in range(len(self.ts)-self.rollout)]

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
            testset = [Data(x=v[i,:,:], fn=fn,
                            y=v[i:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(all_ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rkm=self.rkm
                            ) for i in range(
                                len(self.ts),len(self.ts)+len(self.test_ts)-self.rollout)]
            testset_rollout = [Data(x=v[i,:,:], fn=fn,
                                y=v[i:i+1+self.test_rollout,:,:].transpose(0,1), 
                                irreps_io=self.irreps_io,
                                ts=(all_ts[i:i+1+self.test_rollout].unsqueeze(0)), 
                                pos=pos, edge_index=edge_index, rkm=self.rkm
                                ) for i in range(
                                len(self.ts),len(self.ts)+len(self.test_ts)-self.test_rollout)]
            self.test_data = testset
            self.test_data_rollout = testset_rollout

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return [DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4),
                DataLoader(self.test_data_rollout, num_workers=4)]