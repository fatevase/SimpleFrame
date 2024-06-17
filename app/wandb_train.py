import wandb
import mmengine
from torch.utils.data import DataLoader
from utils.config import Config
from mmengine import MODELS, DATASETS
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner
import app

def train(econfig, project_name='macbook', count=15):
    # Initialize a new wandb run
    wconfig = {
        'method': 'random',
        'metric' : {
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'optimizer': {
                'values': ['AdamW', 'SGD']
            },
            'learning_rate': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': 0,
                'max': 1e-3
            },
            # 'activate': {
            #     'values': ['relu', 'ftanh']
            # },
            'batch_size': {
                # integers between 10 and 25
                # with evenly-distributed logarithms 
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 10,
                'max': 25,
            },
            'epochs': {
                'value': 1
            },
        },
    }

    def core_train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            all_cfg = Config(filename=econfig)
            all_cfg.Optimizer.type = config.optimizer
            all_cfg.Optimizer.lr = config.learning_rate
            all_cfg.train_cfg.max_epochs = config.epochs
            all_cfg.work_dir = './wandb_hp/train_cifar10'
            batch_size = config.batch_size
            app.train(config, batch_size, override_cfg=all_cfg)
            

    wandb.login()
    sweep_id = wandb.sweep(wconfig, project=project_name)
    wandb.agent(sweep_id, core_train, count=count)
