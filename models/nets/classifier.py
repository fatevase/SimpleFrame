# 准备训练任务所需要的模块
from typing import Dict
import torch
import numpy as np
import math
from torch import nn
from mmengine.model import BaseModel
from torch.nn import functional as F
from mmengine import MODELS
from torch.nn.init import xavier_uniform_
from mmengine.visualization import Visualizer

from models.backbone.vit import VitFeature

class FTanh(nn.Module):
    def __init__(self, p=False):
        super().__init__()
        self.alpha = 1.0
        self.beta = 1.0
        if p:
            self.alpha = nn.Parameter(torch.FloatTensor(1))
            self.beta = nn.Parameter(torch.FloatTensor(1))
        self.eps = 1e-8
    def forward(self, x):
        # o=clamp(x⋅( α *1+ϵ),−1,1)
        o = torch.clamp(x*(self.alpha+1e-10), -self.beta, self.beta)
        return o

@MODELS.register_module()
class Classifier(BaseModel):
    def __init__(self, in_channels=1, classify=2, activate='relu', loss='cross_entropy'):
        super().__init__()
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.Conv2d(32, 64, 3, 2, 1),
        ) # scale 1/4
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        if activate == 'relu':
            self.cnn_encoder.add_module('relu', nn.ReLU())
            self.cnn_decoder.add_module('relu', nn.ReLU())
        elif activate == 'ftanh':
            self.cnn_encoder.add_module('ftanh', FTanh())
            self.cnn_decoder.add_module('ftanh', FTanh())
        else:
            self.cnn_encoder.add_module('tanh', nn.Tanh())
            self.cnn_decoder.add_module('tanh', nn.Tanh())
        self.cnn_encoder.add_module('conv_encoder_last', nn.Conv2d(64, 64, 3, 1, 1))
        self.cnn_decoder.add_module('conv_decoder_last', nn.Conv2d(128, 128, 3, 1, 1))
        self.cnn_decoder.add_module('flatten', nn.Flatten(start_dim=1))
        self.cnn_decoder.add_module('fc', nn.Linear(128*4, classify))

        self.vit = VitFeature(out_channels=classify, num_features=classify, vit_model='vit_base_patch16_224', pretrained=True)
        
        self.loss = nn.CrossEntropyLoss() if loss == 'cross_entropy' else nn.MSELoss()

    def forward(self, input: torch.Tensor, data_samples: Dict = dict(), mode: str = 'tensor'):

        x = self.cnn_encoder(input)
        x = self.cnn_decoder(x)
        # x = self.vit(input)


        if mode == 'loss':
            return {'loss': self.loss(x, data_samples['target'])}
        elif mode == 'predict':
            return x.argmax(1)
        else:
            return 
