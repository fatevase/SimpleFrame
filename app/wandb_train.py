import wandb
import mmengine
from torch.utils.data import DataLoader
from utils.config import Config
from mmengine import MODELS, DATASETS
from mmengine.evaluator import Evaluator
from mmengine.runner import Runner

def train(project_name='macbook', count=15):
    # Initialize a new wandb run
    config = {
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
                'max': 0.1
            },
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
            all_cfg = Config(filename='./Python/SimpleFrame/configs/cifar10.py')
            all_cfg.Optimizer.type = config.optimizer
            all_cfg.Optimizer.lr = config.learning_rate
            all_cfg.train_cfg.max_epochs = config.epochs
            batch_size = config.batch_size

            model = mmengine.build_from_cfg(all_cfg.Net, MODELS)

            # 构建手写数字识别 (MNIST) 数据集
            train_dataset = mmengine.build_from_cfg(all_cfg.TDataset, DATASETS)
            val_dataset = mmengine.build_from_cfg(all_cfg.VDataset, DATASETS)

            # 构建数据加载器
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2)

            # 构建评估器
            val_evaluator = Evaluator(all_cfg.Metric)

            # 初始化执行器
            runner = Runner(model,
                            work_dir='./wandb_hp/train_cifar10',  # 工作目录，用于保存模型和日志
                            train_cfg=all_cfg.train_cfg,  # 训练配置
                            visualizer=all_cfg.Visbackend, # 可视化配置
                            train_dataloader=train_dataloader,  # 训练数据加载器
                            val_dataloader=val_dataloader,  # 验证数据加载器
                            val_evaluator=val_evaluator,  # 验证评估器
                            val_cfg=dict(), # 验证配置
                            optim_wrapper=dict(optimizer=all_cfg.Optimizer), # 优化器
                            custom_hooks=[dict(type='ModelVisHook')], # 自定义钩子
                            param_scheduler=all_cfg.Scheduler) # 学习率调度器
            # 执行训练
            runner.train()
            

    wandb.login()
    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id, core_train, count=count)
