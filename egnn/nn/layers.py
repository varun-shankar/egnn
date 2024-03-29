import torch
from torch_scatter import scatter
from e3nn import o3, nn
from torch.nn import ReLU, Linear
from .utils import *
from torch_geometric.nn import GCNConv

## Message Passing Layers ##
class Eq_NLMP(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output,
                       irreps_val='16x0e+4x1o',
                       irreps_fe=o3.Irreps.spherical_harmonics(lmax=2),
                       num_fes=16, hx=4, residual=True, 
                       return_array=False, **kwargs):
        super(Eq_NLMP, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_val = o3.Irreps(irreps_val)
        self.irreps_fe = o3.Irreps(irreps_fe)
        if irreps_input == irreps_output:
            self.residual = residual
        else:
            self.residual = False
        self.return_array = return_array
        
        self.edge_val = LinNet(3*self.irreps_input, hx*self.irreps_input, self.irreps_val, True, **kwargs)

        self.tp = o3.FullyConnectedTensorProduct(self.irreps_val, self.irreps_fe, self.irreps_output, shared_weights=False)
        self.fc = nn.FullyConnectedNet([num_fes, hx*num_fes, self.tp.weight_numel], kwargs.get('act', ReLU()))
        # self.edge_upd = LinNet(self.tp.irreps_out+3*self.irreps_input, hx*self.irreps_output, self.irreps_output, True, **kwargs)

        self.node_upd = LinNet(self.irreps_input+self.irreps_output, hx*self.irreps_output, self.irreps_output, True, **kwargs)

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        v = self.edge_val(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        
        tp = self.tp(v, data.fe, self.fc(data.fes))
        heu = tp#self.edge_upd(torch.cat([tp,data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        hen = (data.he + heu) if self.residual else heu

        node_tmp = scatter(hen*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        hnu = self.node_upd(torch.cat([data.hn,node_tmp],dim=1))
        hnn = (data.hn + hnu) if self.residual else hnu
        
        if self.return_array:
            return hnn, hen
        else:
            data.he = hen; data.hn = hnn
            return data


class nEq_NLMP(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output,
                       irreps_val='4x0e+4x1o',
                       irreps_fe=o3.Irreps.spherical_harmonics(lmax=2),
                       num_fes=16, hx=4, residual=True, 
                       return_array=False, **kwargs):
        super(nEq_NLMP, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_val = o3.Irreps(irreps_val)
        self.irreps_fe = o3.Irreps(irreps_fe)
        if irreps_input == irreps_output:
            self.residual = residual
        else:
            self.residual = False
        self.return_array = return_array
        
        self.edge_val = LinNet(3*self.irreps_input, hx*self.irreps_input, self.irreps_val, False, **kwargs)

        # self.tp = Linear(self.irreps_val.dim+self.irreps_fe.dim+num_fes, self.irreps_output.dim)
        self.fc = nn.FullyConnectedNet([6+num_fes, hx*num_fes, self.irreps_val.dim*self.irreps_output.dim], kwargs.get('act', ReLU()))
        # self.edge_upd = LinNet(self.irreps_output+3*self.irreps_input, hx*self.irreps_output, self.irreps_output, False, **kwargs)

        self.node_upd = LinNet(self.irreps_input+self.irreps_output, hx*self.irreps_output, self.irreps_output, False, **kwargs)

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        v = self.edge_val(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        
        # tp = self.tp(torch.cat([v, data.fe, data.fes],dim=1))
        tp = torch.bmm(v.unsqueeze(1),self.fc(torch.cat([data.pos[edge_src],data.pos[edge_dst], data.fes],dim=1)).reshape(v.shape[0],self.irreps_val.dim,self.irreps_output.dim)).squeeze()
        heu = tp#self.edge_upd(torch.cat([tp,data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        hen = (data.he + heu) if self.residual else heu

        node_tmp = scatter(hen*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        hnu = self.node_upd(torch.cat([data.hn,node_tmp],dim=1))
        hnn = (data.hn + hnu) if self.residual else hnu
        
        if self.return_array:
            return hnn, hen
        else:
            data.he = hen; data.hn = hnn
            return data


class nEq_NLMP_iso(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output,
                       irreps_val='4x0e+4x1o',
                       irreps_fe=o3.Irreps.spherical_harmonics(lmax=2),
                       num_fes=16, hx=4, residual=True, 
                       return_array=False, **kwargs):
        super(nEq_NLMP_iso, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        self.irreps_val = o3.Irreps(irreps_val)
        self.irreps_fe = o3.Irreps(irreps_fe)
        if irreps_input == irreps_output:
            self.residual = residual
        else:
            self.residual = False
        self.return_array = return_array

        self.edge_val = LinNet(3*self.irreps_input, hx*self.irreps_input, self.irreps_val, False, **kwargs)

        # self.tp = Linear(self.irreps_val.dim+num_fes, self.irreps_output.dim)
        self.fc = nn.FullyConnectedNet([num_fes, hx*num_fes, self.irreps_val.dim*self.irreps_output.dim], kwargs.get('act', ReLU()))
        # self.edge_upd = LinNet(self.irreps_output+3*self.irreps_input, hx*self.irreps_output, self.irreps_output, False, **kwargs)

        self.node_upd = LinNet(self.irreps_input+self.irreps_output, hx*self.irreps_output, self.irreps_output, False, **kwargs)

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        v = self.edge_val(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        
        # tp = self.tp(torch.cat([v, data.fes],dim=1))
        tp = torch.bmm(v.unsqueeze(1),self.fc(data.fes).reshape(v.shape[0],self.irreps_val.dim,self.irreps_output.dim)).squeeze()
        heu = tp#self.edge_upd(torch.cat([tp,data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        hen = (data.he + heu) if self.residual else heu

        node_tmp = scatter(hen*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        hnu = self.node_upd(torch.cat([data.hn,node_tmp],dim=1))
        hnn = (data.hn + hnu) if self.residual else hnu
        
        if self.return_array:
            return hnn, hen
        else:
            data.he = hen; data.hn = hnn
            return data




class GCN(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=ReLU(), **kwargs):
        super(GCN, self).__init__()

        self.irreps_input = irreps_input.dim
        self.irreps_output = irreps_output.dim
        self.act = act
        self.f = GCNConv(self.irreps_input, self.irreps_output)

    def forward(self, data):
        data.hn = self.act(self.f(data.hn, data.edge_index))+0*data.he.mean()
        return data 