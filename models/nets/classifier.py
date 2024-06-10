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


@MODELS.register_module()
class Classifier(BaseModel):
    def __init__(self, in_channels=1, classify=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(28 * 28*in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, classify),
            nn.Softmax(dim=1),
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size = 3, stride = 2, padding =  1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(16*32, 200),
            nn.ReLU(),
            nn.Linear(200, classify),
        )

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
        ) # scale 1/4

        self.cnn_decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(128*4, classify)
        )
        self.matrix_h_t = nn.Parameter(torch.FloatTensor(64,1, 8))
        self.matrix_h_f = nn.Parameter(torch.FloatTensor(64,8, 1))
        self.matrix_w_t = nn.Parameter(torch.FloatTensor(64,1, 8))
        self.matrix_w_f = nn.Parameter(torch.FloatTensor(64,8, 1))


        self.channel_matrix = nn.Parameter(torch.FloatTensor(32, in_channels))
        self.w_matrix = nn.Parameter(torch.FloatTensor(28, 14))
        self.h_matrix = nn.Parameter(torch.FloatTensor(28, 14))
        self.scale_pool = nn.AvgPool2d(2 ,2)
        self.conv3x3 = nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.finnally_nn = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16*32, 200),
            nn.ReLU(),
            nn.Linear(200, classify),
        )
        self.bn2d = nn.BatchNorm2d(32)
        self.loss = nn.CrossEntropyLoss()
        self.initialize()
        self.rnn = nn.RNN(8*8, 128, 1, batch_first=True)
        self.flatten = nn.Flatten(start_dim=2)
        self.rnn_classify = nn.Sequential(
            nn.Linear(128, classify),
            nn.Softmax(dim=1),
        )
    def initialize(self):
            xavier_uniform_(self.channel_matrix)
            xavier_uniform_(self.w_matrix)
            xavier_uniform_(self.h_matrix)
            xavier_uniform_(self.matrix_h_t)
            xavier_uniform_(self.matrix_w_t)
            xavier_uniform_(self.matrix_h_f)
            xavier_uniform_(self.matrix_w_f)

    def forward(self, input: torch.Tensor, data_samples: Dict = dict(), mode: str = 'tensor'):
        # x = self.mlp(input)
        # x = self.cnn(input)
        x = self.cnn_encoder(input)
        # x, h = self.rnn(self.flatten(x))
        # x = self.rnn_classify(h.squeeze(0))
        # x = torch.einsum("cnh,bchw->bcnw", self.matrix_h_t, x) / (math.sqrt(8) + 1e-10)
        # x = torch.einsum("chn,bcnw->bchw", self.matrix_h_f, x) / (math.sqrt(8) + 1e-10)
        # x = torch.einsum("cnw,bchw->bcnh", self.matrix_w_t, x) / (math.sqrt(8) + 1e-10)
        # x = torch.einsum("cwn,bcnh->bchw", self.matrix_w_f, x) / (math.sqrt(8) + 1e-10)
        x = self.cnn_decoder(x)
        # x = torch.einsum("kc,bchw->bkhw", self.channel_matrix, input) / (math.sqrt(32*1) + 1e-10)
        # x = torch.einsum("hi,bkhw->bkiw", self.w_matrix, x) / (math.sqrt(28*28) + 1e-10)
        # x = torch.einsum("hj,bkiw->bkij", self.h_matrix, x) / (math.sqrt(28*28) + 1e-10)
        # x = self.scale_pool(x)
        # x = self.finnally_nn(x)


        if mode == 'loss':
            return {'loss': self.loss(x, data_samples['target'])}
        elif mode == 'predict':
            return x.argmax(1)
        else:
            return 
