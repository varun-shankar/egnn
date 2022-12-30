import torch
import copy, sys
from e3nn import o3
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
import pytorch_lightning as pl
from torchmetrics import Metric

from egnn.nn.subnets import *
from egnn.nn.layers import *
ACT_DICT = {'tanh': Tanh(), 'relu': ReLU(), 'elu': ELU(), 'silu': SiLU(), 'sigmoid': Sigmoid()}

## Model ##
class Dudt(torch.nn.Module):
    def __init__(self, enc, f, dec, num_levels=1, cf=[], **kwargs):
        super(Dudt, self).__init__()

        self.num_levels = num_levels
        self.enc = enc
        self.gps = torch.nn.ModuleList()
        for i in range(num_levels-1): 
            self.gps.append(GraphPool(cf[i], f.layer_type, f.irreps_latent, **kwargs))
        self.f = f
        self.dec = dec
        self.data = None

    def forward(self, t, u):
        self.data.hn = u

        self.data = self.data.resample_edges(self.data.rkm[0][0])
        self.data = self.enc(self.data)

        self.data = [self.data]
        for i in range(self.num_levels-1):
            self.data.append(self.gps[i](self.data[i], self.data[0].rkm[0][i+1]))

        data = self.f(self.data)
        data[0].sg_idx.append([data[i].idx for i in range(1,len(data))])
        self.data = data[0]

        self.data = self.dec(self.data)
        return self.data.hn

def build_model(model_type, irreps_io,
                latent_layers, latent_scalars, latent_vectors=0,
                irreps_hidden=None, **kwargs):
    if 'act' in kwargs:
        kwargs['act'] = ACT_DICT[kwargs['act']]
    irreps_in = o3.Irreps(irreps_io[2])+o3.Irreps(irreps_io[0])
    irreps_latent = o3.Irreps(f'{latent_scalars:g}'+'x0e + '+f'{latent_vectors:g}'+'x1o')
    irreps_out = o3.Irreps(irreps_io[1])
    irreps_hidden = '4x0e+4x1o' if irreps_hidden is None else irreps_hidden

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

    enc = Encoder(irreps_in, irreps_hidden, irreps_latent, model_type, **kwargs)
    f = Latent(layer_type, irreps_latent, latent_layers, **kwargs)
    dec = Decoder(irreps_latent, irreps_hidden, irreps_out, model_type, **kwargs)
    dudt = Dudt(enc, f, dec, **kwargs)
    model = NODE(dudt, **kwargs)

    return model

class RunningVar(Metric):
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

class LitModel(pl.LightningModule):
    def __init__(self, dm,
                latent_layers=4, latent_scalars=8, latent_vectors=8, 
                model_type='equivariant', noise_var=0, noise_fac=0, data_aug=False,
                lr=1e-3, epochs=None, lr_sch=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.node = build_model(model_type, dm.irreps_io,
            latent_layers, latent_scalars, latent_vectors, **kwargs)
        self.loss_fn = dm.loss_fn
        self.lr = lr
        self.lr_sch = lr_sch
        self.epochs = epochs
        self.noise_var = noise_var
        self.noise_fac = noise_fac
        self.var = RunningVar()
        self.data_aug = data_aug

    def forward(self, data):

        data.sg_idx = []
        _ = data.rotate(o3.rand_matrix()) if self.data_aug and self.training else 0

        xi = data.x
        if self.training:
            xi = xi + o3.Irreps(data.irreps_io[0][0]).randn(xi.shape[0],-1).type_as(xi) * \
            (self.noise_var)**.5
        
        self.node.dudt.data = data
        t, yhs = self.node(xi, data.ts[0,:])

        return yhs[1:,:,:]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch:
            lr_sch = OneCycleLR(optimizer, self.lr, self.epochs, pct_start=0.1)
            # lr_sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
            # lr_sch2 = StepLR(optimizer, step_size=20, gamma=0.1)
            return [optimizer], [lr_sch]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        data = batch#.subsample()
        y_hat = self(data)
        self.var.update(y_hat[-1], data.y.transpose(0,1)[-1])

        loss_dict = self.loss_fn(y_hat, data)
        for k, v in loss_dict.items():
            self.log('train_'+k, v, batch_size=data.num_graphs)
        return loss_dict['loss']
    
    def training_epoch_end(self, outputs):
        frac = self.noise_fac
        self.noise_var = self.noise_var * (1-frac) + self.var.compute() * frac 
        self.var.reset()

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)

        loss_dict = self.loss_fn(y_hat, data)
        for k, v in loss_dict.items():
            self.log('val_'+k, v, batch_size=data.num_graphs)
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
                    add_dataloader_idx=False)

            eq_loss = torch.nn.functional.mse_loss(y_hat @ D_out.T, y_hat_rot)
            self.log('eq_loss', eq_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'pred.pt')
        else:
            data = batch
            y_hat = self(data)
            loss_dict = self.loss_fn(y_hat, data)
            for k, v in loss_dict.items():
                self.log('test_rollout_'+k, v, batch_size=data.num_graphs,
                    add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'pred_rollout.pt')

        return loss_dict['loss']
