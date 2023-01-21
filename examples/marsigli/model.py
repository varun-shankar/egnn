import copy, math
import torch
from torch.nn import Tanh, ReLU, ELU, SiLU, Sigmoid
from e3nn import o3
from egnn.nn.subnets import *
from egnn.nn.layers import *
from torchdyn.core import NeuralODE
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR, LambdaLR
import pytorch_lightning as pl
from torchmetrics import Metric
ACT_DICT = {'tanh': Tanh(), 'relu': ReLU(), 'elu': ELU(), 'silu': SiLU(), 'sigmoid': Sigmoid()}

## Model ##
class Dudt(torch.nn.Module):
    def __init__(self, enc, lat, dec, num_levels=1, cf=[], **kwargs):
        super(Dudt, self).__init__()

        self.num_levels = num_levels
        self.enc = enc
        self.gps = torch.nn.ModuleList()
        for i in range(num_levels-1): 
            self.gps.append(GraphPool(cf[i], lat.layer_type, lat.irreps_latent, **kwargs))
        self.lat = lat
        self.dec = dec
        self.data = None

    def forward(self, t, u):
        self.data.hn = u

        self.data = self.data.resample_edges(self.data.rkm[0][0], self.training)
        self.data = self.enc(self.data)

        self.data = [self.data]
        for i in range(self.num_levels-1):
            self.data.append(self.gps[i](self.data[i], self.data[0].rkm[0][i+1]))

        data = self.lat(self.data)
        data[0].sg_idx.append([data[i].idx for i in range(1,len(data))])
        self.data = data[0]

        self.data = self.dec(self.data)
        return self.data.hn

class GraphNet(torch.nn.Module):
    def __init__(self, irreps_io,
                model_type='equivariant', latent_layers=4, latent_dim=64, 
                latent_vectors=False, irreps_hidden=None, 
                solver='euler', sensitivity='autograd', **kwargs):
        super(GraphNet, self).__init__()
        if 'act' in kwargs:
            kwargs['act'] = ACT_DICT[kwargs['act']]
        if latent_vectors:
            latent_scalars = int(latent_dim/4)
            latent_vectors = int(latent_dim/4)
        else:
            latent_scalars = latent_dim
            latent_vectors = 0

        irreps_fn = o3.Irreps(irreps_io[2])
        irreps_in = o3.Irreps(irreps_io[0])
        irreps_latent = o3.Irreps(f'{latent_scalars:g}'+'x0e + '+f'{latent_vectors:g}'+'x1o').simplify()
        irreps_out = o3.Irreps(irreps_io[1])
        irreps_hidden = (4*(irreps_fn+irreps_in)).sort().irreps.simplify() if irreps_hidden is None else irreps_hidden

        if model_type == 'equivariant':
            if latent_vectors == 0:
                layer_type = nEq_NLMP_iso
            else:
                layer_type = Eq_NLMP
        elif model_type == 'non-equivariant':
            layer_type = nEq_NLMP
        elif model_type == 'non-equivariant isotropic':
            layer_type = nEq_NLMP_iso
        elif model_type == 'GCN':
            layer_type = GCN
        else:
            print('Unknown model type: ', model_type)

        self.enc = Encoder(irreps_fn, irreps_in, irreps_hidden, irreps_latent, model_type, **kwargs)
        self.lat = Latent(layer_type, irreps_latent, latent_layers, **kwargs)
        self.dec = Decoder(irreps_latent, irreps_hidden, irreps_out, model_type, **kwargs)
        
        # for p in self.enc.parameters(): p.requires_grad = False
        # for p in self.dec.parameters(): p.requires_grad = False

        self.dudt = Dudt(self.enc, self.lat, self.dec, **kwargs)
        with torch.no_grad():
            self.ode = NeuralODE(self.dudt, sensitivity=sensitivity, solver=solver)

    def forward(self, u0, data):

        data.sg_idx = []
        self.dudt.data = data
        t, yhs = self.ode(u0, data.ts[0,:])
        # self.data = data
        # self.data.hn = u0
        # self.data = self.data.resample_edges(self.data.rkm[0][0], self.training)
        # self.data = self.enc(self.data)
        # self.data = self.dec(self.data)
        # yhs = self.data.hn.unsqueeze(0)


        return yhs

class LitModel(pl.LightningModule):
    def __init__(self, irreps_io,
                 epochs=None, data_aug=False,
                 lr=1e-3, lr_sch=False, 
                 noise_fac=0, noise_sch=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.mod = GraphNet(irreps_io, **kwargs)
        self.epochs = epochs
        self.lr = lr
        self.lr_sch = lr_sch
        self.noise_var = 0
        self.noise_fac = noise_fac
        self.noise_sch = noise_sch
        self.var = RunningVar()
        self.data_aug = data_aug

    def forward(self, data):

        _ = data.rotate(o3.rand_matrix()) if self.data_aug and self.training else 0

        xi = data.x
        if self.training:
            xi = xi + o3.Irreps(data.irreps_io[0][0]).randn(xi.shape[0],-1).type_as(xi) * \
            (self.noise_var)**.5

        return self.mod(xi, data)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch:
            lr_sch = LambdaLR(optimizer, lr_lambda=(lambda epoch: 9*math.exp(-10/self.epochs*epoch)+1))
            # lr_sch = OneCycleLR(optimizer, self.lr, self.epochs, pct_start=0.33)
            # lr_sch = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
            # lr_sch = StepLR(optimizer, step_size=20, gamma=0.1)
            return [optimizer], [lr_sch]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        data = batch#.subsample()
        y_hat = self(data)

        loss_dict = self.loss_fn(y_hat, data)
        for k, v in loss_dict.items():
            self.log('train_'+k, v, batch_size=data.num_graphs)
        return loss_dict['loss']
    
    def training_epoch_end(self, outputs):
        L =  self.noise_fac; k = -30/self.epochs; x0 = 0.33*self.epochs
        multiplier = L/(1+math.exp(k*(self.current_epoch-x0))) if self.noise_sch else L
        self.noise_var = multiplier #* self.var.compute()
        self.log('noise_multiplier', multiplier, sync_dist=True)

        self.var.reset()

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        self.var.update(y_hat[-1], data.y.transpose(0,1)[-1])

        loss_dict = self.loss_fn(y_hat, data)
        for k, v in loss_dict.items():
            self.log('val_'+k, v, batch_size=data.num_graphs, sync_dist=True)
        return loss_dict['loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            data = batch

            rot_data = copy.deepcopy(data)
            rot = o3.rand_matrix()
            rot_data, D_out = rot_data.rotate(rot)

            y_hat = self(data)
            y_hat_rot = self(rot_data)

            loss_dict = self.loss_fn(y_hat, data)
            for k, v in loss_dict.items():
                self.log('test_'+k, v, batch_size=data.num_graphs,
                    add_dataloader_idx=False, sync_dist=True)

            eq_loss = torch.nn.functional.mse_loss(y_hat @ D_out.T, y_hat_rot)
            self.log('eq_loss', eq_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False, sync_dist=True)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'pred.pt')
        else:
            data = batch
            y_hat = self(data)
            loss_dict = self.loss_fn(y_hat, data)
            for k, v in loss_dict.items():
                self.log('test_rollout_'+k, v, batch_size=data.num_graphs,
                    add_dataloader_idx=False, sync_dist=True)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'pred_rollout.pt')

        return loss_dict['loss']

    def loss_fn(self, y_hat, data):
        yh = y_hat[1:,:,:]; yt = data.y.transpose(0,1)[1:,:,:]
        dict = {'loss': torch.mean((yh - yt)**2),
                'T_loss': torch.mean((yh[:,:,-1] - yt[:,:,-1])**2)}
        return dict

####################################################################################################

class RunningVar(Metric):
    full_state_update: bool = True
    def __init__(self):
        super().__init__()
        self.add_state("var", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.var = self.var + torch.mean((preds - target)**2, dim=0, keepdim=True)
        self.total += 1

    def compute(self):
        return self.var / self.total
