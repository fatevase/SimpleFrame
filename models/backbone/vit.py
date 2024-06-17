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
import timm


@MODELS.register_module()
class VitFeature(BaseModel):
    def __init__(self, out_channels, num_features, vit_model='vit_base_patch16_224', pretrained=True):
        super(VitFeature, self).__init__()
        # 加载预训练的ViT模型
        self.vit_model = timm.create_model(vit_model, pretrained=pretrained)
        # 冻结模型中的所有权重
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        # 替换ViT的分类头部以用于特征提取
        self.vit_model.head = nn.Identity()

        # 添加自定义的特征提取层
        self.feature_extractor = nn.Linear(self.vit_model.num_features, num_features)
        self.output_layer = nn.Linear(num_features, out_channels)

    def forward(self, x):
        # 通过ViT模型提取特征
        features = self.vit_model(x)

        # 应用自定义的特征提取层
        features = self.feature_extractor(features)
        output = self.output_layer(features)
        return features