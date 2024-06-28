import wandb
import mmengine
from torch.utils.data import DataLoader
from utils.config import Config
from mmengine import MODELS, DATASETS
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner
import app
import time
def train(econfig, project_name='macbook', count=200):
    # Initialize a new wandb run
    wconfig = {
        "name": f"sweep-{time.time()}",
        'method': 'bayes',
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
                'max': 1e-2
            },
            'embad_type':{
                'values': ['conv', 'embed']
            },
            'attention_shape': {
                'values': [
                    (2, 2), (4, 4), (16, 16), (32, 32)]
            },
            'attention_in': {
                'values': [64, 128, 256]
            },
            'attention_times': {
                'values': [1,2,3,4]
            },
            'multi_head': {
                'values': [1,2,4]
            },
            # 'batch_size': {
            #     # integers between 10 and 25
            #     # with evenly-distributed logarithms 
            #     'distribution': 'q_log_uniform_values',
            #     'q': 8,
            #     'min': 10,
            #     'max': 25,
            # },
            'epochs': {
                'value': 3
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
            
            all_cfg.Net.embad_type = config.embad_type
            all_cfg.Net.attention_in = config.attention_in
            all_cfg.Net.attention_times = config.attention_times
            all_cfg.Net.multi_head = config.multi_head
            
            # batch_size = config.batch_size
            app.train("", override_cfg=all_cfg)
            

    wandb.login()
    sweep_id = wandb.sweep(wconfig, project=project_name)
    wandb.agent(sweep_id, core_train, count=count)
