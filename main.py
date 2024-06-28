# from app.wandb_train import train
import argparse
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
    args = argparse.ArgumentParser()
    args.add_argument('--train', action='store_true')
    args.add_argument('--val', action='store_true')
    args.add_argument('--load_from', type=str, default='')
    args.add_argument('--data_type', type=str, default='val')
    args = args.parse_args()
    
    seed_everything()
    if args.train:
        train('./configs/cifar10.py')
    if args.val and args.load_from != '':
        val('./configs/cifar10.py', data_type=args.data_type, load_from=args.load_from)