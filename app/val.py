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

def val(config_path: str, data_type='val', load_from=None, batch_size=40, num_workers=2):

    
    all_cfg = Config(filename=config_path)

    model = mmengine.build_from_cfg(all_cfg.Net, MODELS)

    # 构建手写数字识别 (MNIST) 数据集
    if data_type == 'train':
        val_dataset = mmengine.build_from_cfg(all_cfg.TDataset, DATASETS)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        val_dataset = mmengine.build_from_cfg(all_cfg.VDataset, DATASETS)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers)

    # 构建评估器
    val_evaluator = Evaluator(all_cfg.Metric)

    # 初始化执行器
    runner = Runner(model,
                    work_dir=all_cfg.work_dir,  # 工作目录，用于保存模型和日志
                    visualizer=all_cfg.Visbackend,  # 可视化后端
                    val_dataloader=val_dataloader,  # 验证数据加载器
                    val_evaluator=val_evaluator,  # 验证评估器
                    val_cfg=dict(), # 验证配置
                    load_from=load_from, # 加载模型
                    optim_wrapper=None, # 优化器
                    custom_hooks=[dict(type='ModelVisHook')]) # 学习率调度器
    # 执行训练
    # runner.train()
    # LoggerHook.after_train_iter()
    runner.val()

    # # 初始化执行器
    # runner = Runner(model=model, test_dataloader=val_dataloader, test_evaluator=val_evaluator, test_cfg=dict(),
    #                 load_from='./train_mnist/epoch_3.pth', work_dir='./test_mnist')
    # # 执行测试
    # runner.test()