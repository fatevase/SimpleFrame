from app.train import train
import wandb

if __name__ == "__main__":
    train('./configs/cifar10.py')