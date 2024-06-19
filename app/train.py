from mmengine.runner import Runner
import cv2
import torch
from torch.utils.data import DataLoader
from mmengine import MODELS, PARAM_SCHEDULERS, OPTIMIZERS, DATASETS, METRICS
import mmengine
from mmengine.evaluator import Evaluator
from utils.config import Config
from evaluator import ClassifyAccuracy
from mmengine.hooks import LoggerHook

# from logging.logger import MMLogger

def train(config_path: str, batch_size=20, num_workers=2, override_cfg=None):

    if override_cfg:
        all_cfg = override_cfg
    else:
        all_cfg = Config(filename=config_path)

    model = mmengine.build_from_cfg(all_cfg.Net, MODELS)
    print(model)


    train_dataset = mmengine.build_from_cfg(all_cfg.TDataset, DATASETS)
    val_dataset = mmengine.build_from_cfg(all_cfg.VDataset, DATASETS)

    # val_dataset = train_dataset.get_subset(int(len(train_dataset) *0.7), len(train_dataset))
    # train_dataset = train_dataset.get_subset(0, int(len(train_dataset) *0.7))

    # 构建数据加载器
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)

    # 构建评估器
    val_evaluator = Evaluator(all_cfg.Metric)

    # 初始化执行器
    runner = Runner(model,
                    work_dir=all_cfg.work_dir,  # 工作目录，用于保存模型和日志
                    visualizer=all_cfg.Visbackend,  # 可视化后端
                    train_cfg=all_cfg.train_cfg,  # 训练配置
                    train_dataloader=train_dataloader,  # 训练数据加载器
                    val_dataloader=val_dataloader,  # 验证数据加载器
                    val_evaluator=val_evaluator,  # 验证评估器
                    val_cfg=dict(), # 验证配置
                    # load_from='/Users/vase/Documents/Coding/Python/train_cifar10/epoch_3.pth',
                    optim_wrapper=dict(optimizer=all_cfg.Optimizer), # 优化器
                    custom_hooks=[dict(type='ModelVisHook'), dict(type='TrainAccuracyHook')],
                    param_scheduler=all_cfg.Scheduler) # 学习率调度器
    # 执行训练
    runner.train()
    # LoggerHook.after_train_iter()
    # runner.val()

    # # 初始化执行器
    # runner = Runner(model=model, test_dataloader=val_dataloader, test_evaluator=val_evaluator, test_cfg=dict(),
    #                 load_from='./train_mnist/epoch_3.pth', work_dir='./test_mnist')
    # # 执行测试
    # runner.test()