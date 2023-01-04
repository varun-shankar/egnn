from data import DataModule
from model import LitModel
import pytorch_lightning as pl
from egnn.learn import *

config, cfg_file = load_config()
dm, model, ckpt, run_id = lightning_setup(config, DataModule, LitModel)

bstrainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, log_every_n_steps=1)
bs = bstrainer.tuner.scale_batch_size(model, datamodule=dm, mode='binsearch')
with open('.batch_size', 'w') as fl:
    fl.write(str(bs))