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
from models.backbone.attension import Attention
from models.backbone.cnn_transformer import CNNTransformer

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
    def __init__(self, in_channels=1, classify=2, attention_in=64, attention_times=1, multi_head=3, embad_type='conv', loss='cross_entropy'):
        super().__init__()
        

        
        self.cnn_trans = CNNTransformer(3, (32, 32), attention_in, (8, 8), 0, multi_head, attention_times, embad_type=embad_type)

        # self.cnn_trans = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 2, 1),
        #     nn.Conv2d(64, attention_in, 3, 2, 1),
        # )
        
        self.tran_decoder = nn.Sequential(
            nn.Conv2d(attention_in, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(64*16, classify)
        )
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        ) # scale 1/4
        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(attention_in, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 2, 1),
        )
        # self.cnn_encoder.add_module('conv_encoder_last', nn.Conv2d(64, 64, 3, 1, 1))
        # # self.cnn_decoder.add_module('conv_decoder_last', nn.Conv2d(128, 128, 3, 1, 1))
        # self.cnn_decoder.add_module('flatten', nn.Flatten(start_dim=1))
        # self.cnn_decoder.add_module('fc', nn.Linear(128*4, classify))
        # self.cnn_decoder.add_module('activate', nn.Softmax())

        
        
        # self.attn_pre = nn.Conv2d(in_channels, attention_in, 3, 1, 1)
        # self.atn = Attention(attention_in, attention_in, attention_times=attention_times, multi_head=multi_head, input_shape=(32, 32))
        
        self.loss = nn.CrossEntropyLoss() if loss == 'cross_entropy' else nn.MSELoss()


    def forward(self, input: torch.Tensor, data_samples: Dict = dict(), mode: str = 'tensor'):
        
        x = self.cnn_trans(input)
        x = self.tran_decoder(x)
        # input = self.attn_pre(input)
        # input = self.atn(input)
        # input = self.attn_post(input)
        # x = self.cnn_encoder(input)
        # x = self.cnn_decoder(input)
        # x = self.vit(input)


        if mode == 'loss':
            return {'loss': self.loss(x, data_samples['target'])}
        elif mode == 'predict':
            return x.argmax(1)
        else:
            return 
