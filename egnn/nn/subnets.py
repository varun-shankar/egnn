import torch
from e3nn import o3
from .layers import *
from .utils import *
from ..data.modules import Data

# Pooling #
class GraphPool(torch.nn.Module):
    def __init__(self, cf, layer_type, irreps_latent, pool_type='random',
                       irreps_fe=o3.Irreps.spherical_harmonics(lmax=2), **kwargs):
        super(GraphPool, self).__init__()
        
        self.pool_type = pool_type
        if self.pool_type == 'topk':
            self.scorer = layer_type(irreps_latent, '0e', return_array=True, **kwargs)
        self.cf = cf
        self.edge_enc = LinNet(irreps_fe, irreps_latent, irreps_latent, layer_type==Eq_NLMP, **kwargs)

    def forward(self, data, r):

        if self.pool_type == 'topk':
            score, _ = self.scorer(data)
            idx = torch.topk(score.flatten(),int(data.num_nodes/self.cf),dim=0).indices
            gate = torch.sigmoid(score[idx])
        elif self.pool_type == 'random':
            idx = torch.randperm(data.num_nodes)[:int(data.num_nodes/self.cf)]; gate = 1

        subdata = Data(hn=data.hn[idx]*gate, pos=data.pos[idx], batch=data.batch[idx], idx=idx)
        subdata = subdata.resample_edges(r)
        subdata.he = self.edge_enc(subdata.fe)

        return subdata

class vWrap(torch.nn.Module):
    def __init__(self, layer_type, irreps_latent, *args, 
                num_levels=1, skip_mp_levels=[], **kwargs):
        super(vWrap, self).__init__()

        self.num_levels = num_levels
        self.layers = torch.nn.ModuleList()
        for i in range(num_levels):
            if i in skip_mp_levels:
                l = torch.nn.Identity()
            else:
                l = layer_type(irreps_latent, irreps_latent, *args, **kwargs)
            self.layers.append(l)

        self.up = torch.nn.ModuleList()
        for i in range(num_levels-1):
            self.up.append(LinNet(2*irreps_latent, irreps_latent, irreps_latent, layer_type==Eq_NLMP, **kwargs))
        
    def forward(self, data_list):

        for i in range(self.num_levels):
            data_list[i] = self.layers[i](data_list[i])

        for i in reversed(range(self.num_levels-1)):
            input = 0*data_list[i].hn; input[data_list[i+1].idx] = data_list[i+1].hn
            data_list[i].hn += self.up[i](
                torch.cat([input,data_list[i].hn],dim=1))

        return data_list

class Latent(torch.nn.Module):
    def __init__(self, layer_type, irreps_latent, num_layers, **kwargs):
        super(Latent, self).__init__()

        self.layer_type = layer_type
        self.irreps_latent = irreps_latent
        self.layers = torch.nn.ModuleList()
        self.layers.append(NormLayer())
        for i in range(num_layers):
            sml = kwargs.get('skip_mp_levels_all', None)
            skip_mp_levels = [] if sml is None else sml[i]
            self.layers.append(vWrap(layer_type, irreps_latent, 
                skip_mp_levels=skip_mp_levels, **kwargs))
            self.layers.append(NormLayer())

    def forward(self, data):

        for i in range(len(self.layers)):
            data = self.layers[i](data)

        return data


# Encoder/Decoder #
class Encoder(torch.nn.Module):
    def __init__(self, irreps_fn, irreps_in, irreps_hidden, irreps_latent, model_type='equivariant', 
                       irreps_fe=o3.Irreps.spherical_harmonics(lmax=2), **kwargs):
        super(Encoder, self).__init__()

        if model_type=='equivariant':
            self.node_proj = o3.Linear(irreps_fn+irreps_in, irreps_hidden)
            self.edge_proj = o3.Linear(irreps_fe, irreps_hidden)
            mp_layer = Eq_NLMP
        else:
            self.node_proj = Linear(irreps_fn.dim+irreps_in.dim, irreps_hidden.dim)
            self.edge_proj = Linear(irreps_fe.dim, irreps_hidden.dim)
            mp_layer = nEq_NLMP

        # self.mp = torch.nn.Sequential(
        #     mp_layer(irreps_hidden, irreps_hidden, **kwargs), NormLayer(),
        #     mp_layer(irreps_hidden, irreps_latent, **kwargs)
        # )
        self.mp = mp_layer(irreps_hidden, irreps_latent, **kwargs)

    def forward(self, data):

        data.hn = self.node_proj(torch.cat([data.fn,data.hn],dim=-1))
        data.he = self.edge_proj(data.fe)
        data = self.mp(data)

        return data

class Decoder(torch.nn.Module):
    def __init__(self, irreps_latent, irreps_hidden, irreps_out, model_type='equivariant', **kwargs):
        super(Decoder, self).__init__()


        if model_type=='equivariant':
            mp_layer = Eq_NLMP
            self.node_proj = o3.Linear(irreps_hidden, irreps_out)
        else:
            mp_layer = nEq_NLMP
            self.node_proj = Linear(irreps_hidden.dim, irreps_out.dim)

        # self.mp = torch.nn.Sequential(
        #     mp_layer(irreps_latent, irreps_hidden, **kwargs), NormLayer(),
        #     mp_layer(irreps_hidden, irreps_hidden, **kwargs)
        # )
        self.mp = mp_layer(irreps_latent, irreps_hidden, **kwargs)

    def forward(self, data):

        data = self.mp(data)
        data.hn = self.node_proj(data.hn)

        return data

