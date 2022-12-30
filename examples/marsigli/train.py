import os
from types import SimpleNamespace
from data import DataModule
from model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from egnn.learn import *

### Config ###
config = load_config()
config['dataset'] = os.path.basename(os.path.dirname(__file__))

## Read data ###
dm = DataModule(**config)

### Lightning setup ###
model, ckpt, run_id = lightning_setup(config, dm, LitModel)

### Train ###
wandb_logger = WandbLogger(project=config.get('project'), log_model=True, 
    job_type=config.get('job_type'), id=run_id, settings=wandb.Settings(start_method="fork"), 
    config=SimpleNamespace(**config))
if config.get('job_type') != 'eval':
    wandb_logger.watch(model)
    checkpoint_callback = ModelCheckpoint(monitor=config.get('monitor','val_loss'), 
        dirpath='checkpoints/', filename='run-'+run_id+'-best')
    lr_monitor = LearningRateMonitor()
    strategy = 'ddp_find_unused_parameters_false' if config.get('gpus') != 1 else None
    trainer = pl.Trainer(
        gpus=config.get('gpus'), strategy=strategy, precision=16,
        logger=wandb_logger, callbacks=[checkpoint_callback,lr_monitor],
        max_epochs=config.get('epochs'), log_every_n_steps=10,
        #resume_from_checkpoint=ckpt, accumulate_grad_batches=2
    )
    trainer.fit(model, dm, ckpt_path=ckpt)
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        trainer = pl.Trainer(gpus=1, logger=wandb_logger, limit_test_batches=5)
        trainer.test(model, datamodule=dm)
else:
    dm.setup('fit')
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, limit_test_batches=5)
    trainer.test(model, datamodule=dm)

from plot import plot
plot('pred_rollout.pt')