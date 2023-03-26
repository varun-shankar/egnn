import os, glob, sys, yaml, argparse
import wandb

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        help='Configuration file', type=str)
    args = parser.parse_args()
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)
    return config, args.config


def lightning_setup(config, DataModule, LitModel, run_id=None, load_opt_state=False):
    dm = DataModule(**config)

    if config.get('job_type') == 'train':
        model = LitModel(**dm.model_args, **config)
        ckpt = None
        run_id = wandb.util.generate_id() if run_id is None else run_id
    elif config.get('job_type') == 'retrain':
        if config.get('load_id', None) is None:
            ckpt = max(glob.glob('checkpoints/*'), key=os.path.getctime)
        else:
            ckpt = max(glob.glob('checkpoints/run-'+config.get('load_id')+'*'), key=os.path.getctime)
        print('Loading '+ckpt)
        model = LitModel.load_from_checkpoint(ckpt, **dm.model_args, **config)
        ckpt = ckpt if load_opt_state else None
        run_id = wandb.util.generate_id() if run_id is None else run_id
    elif config.get('job_type') in {'resume', 'eval'}:
        if config.get('load_id', None) is None:
            ckpt = max(glob.glob('checkpoints/*'), key=os.path.getctime)
        else:
            ckpt = max(glob.glob('checkpoints/run-'+config.get('load_id')+'*'), key=os.path.getctime)
        print('Loading '+ckpt)
        model = LitModel.load_from_checkpoint(ckpt, **dm.model_args, **config)
        import re
        run_id = re.search('run-(.*)-last', ckpt).group(1)
    else:
        print('Unknown job type')

    return dm, model, ckpt, run_id