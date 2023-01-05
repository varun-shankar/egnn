import os
from types import SimpleNamespace
from data import DataModule
from model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Timer, RichProgressBar, RichModelSummary, StochasticWeightAveraging
import wandb
from egnn.learn import *

### Config ###
config, cfg_file = load_config()
config['dataset'] = os.path.basename(os.path.dirname(__file__))

# Sweep
wandb.init(config=SimpleNamespace(**config))
config = wandb.config
if config.get('model_type') == 'non-equivariant':
    config.update({'lr': config['lr']/5}, allow_val_change=True)
if config.get('latent_vectors'):
    config.update({'batch_size': int(config['batch_size']/2)}, allow_val_change=True)

### Lightning setup ###
dm, model, ckpt, run_id = lightning_setup(config, DataModule, LitModel)

### Train ###
wandb_logger = WandbLogger(project=config.get('project'), log_model=True, 
    job_type=config.get('job_type'), id=run_id, settings=wandb.Settings(start_method="fork"), 
    config=SimpleNamespace(**config), resume="allow")

# Callbacks
rpb = RichProgressBar(), RichModelSummary(max_depth=3)
checkpoint_callback = ModelCheckpoint(monitor=config.get('monitor','val_loss'), 
        dirpath='checkpoints/', filename='run-'+run_id+'-best')
lr_monitor = LearningRateMonitor(logging_interval='epoch')
early_stopping = EarlyStopping('val_loss', patience=1e3, stopping_threshold=1e-7)
timer = Timer(duration="00:05:00:00")
swa = StochasticWeightAveraging(config.get('lr'), swa_epoch_start=0.95, annealing_epochs=int(.005*config.get('epochs')))
callbacks = [*rpb,checkpoint_callback,lr_monitor,swa,early_stopping]

# Fit/eval
if config.get('job_type') != 'eval':
    wandb_logger.watch(model.mod)
    strategy = 'ddp_find_unused_parameters_false' if config.get('gpus') != 1 else None
    trainer = pl.Trainer(accelerator='gpu',
        devices=config.get('gpus'), strategy=strategy, precision=16,
        logger=wandb_logger, callbacks=callbacks, limit_test_batches=10,
        max_epochs=config.get('epochs'), log_every_n_steps=5
    )
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt)
    trainer.test(model, datamodule=dm)
else:
    dm.setup('fit')
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=callbacks, 
        logger=wandb_logger, limit_test_batches=10)
    trainer.test(model, datamodule=dm)

### Plotting ###
if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    from plot import plot
    print('Plotting...')
    plot('pred_rollout.pt')