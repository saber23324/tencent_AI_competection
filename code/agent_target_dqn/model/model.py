#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import List
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.conf.conf import  args

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
      nn.init.constant_(layer.bias, bias_const)
  return layer
  
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        self.feature_len = Config.DIM_OF_OBSERVATION - Config.FEATURE_SPLIT_SHAPE[-1]
        # Q network
        # Q 网络
        self.q_mlp = MLP([self.feature_len, 256, 128, action_shape], "q_mlp")
        self.backbone = Backbone()

    def reshaped_forward(self, feature):
         # 假设 feature 的形状为 (batch_size, self.feature_len)
        batch_size = feature.shape[0]
        
        # 分割 feature，前 self.feature_len - 121 维度作为常规特征
        obs_map_backbone = feature[:, :args.DIM_OF_BACKBONE]  # (batch_size, self.feature_len - 10404)(4, 51, 51)
        
        # 提取后 121 维度并 reshape 为卷积所需格式 (batch_size, 4, 51, 51)
        regular_feature = feature[:,args.DIM_OF_BACKBONE:]  # (batch_size, 10404)

        return regular_feature, obs_map_backbone
    # Forward inference
    # 前向推理
    def forward(self, feature):
        # Action and value processing
        regular_feature, obs_map_backbone = self.reshaped_forward(feature)
        backbone_out = self.backbone(obs_map_backbone) #512
        combined_feature = torch.cat([regular_feature,backbone_out], -1)
        # 输入到 MLP
        logits = self.q_mlp(combined_feature)
        return logits

        

def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=7, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4096, 512),
        nn.ReLU()
        )
        self.fc = nn.Sequential(
        nn.Linear(512+args.observation_vec_shape, 512),
        nn.ReLU()
        )
        self.apply(layer_init)
    def forward(self, x):
        obs_size = np.prod(args.observation_img_shape)
        B = x.shape[0]
        img, vec = x[:, :obs_size], x[:, obs_size:]
        img = img.view(B, *args.observation_img_shape)
        x = torch.cat([self.cnn(img), vec], -1)
        return self.fc(x)