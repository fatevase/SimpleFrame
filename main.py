# from app.wandb_train import train
from app.train import train
from app.val import val
import wandb

import random
import torch
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seed_everything()
    train('./configs/cifar10.py')
    # val('./configs/cifar10.py', 'val', load_from='D:/Codes/Python/SimpleFrame/logs/train_cifar10/epoch_10.pth')