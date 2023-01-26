    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:47:29 2022

@author: tan
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                           ('conv1', nn.Conv2d(1, 10, kernel_size=3)),
                           ('pool1', nn.MaxPool2d(3))
                           ]))
        
        self.classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(640, 50)),
                           ('relu3', nn.ReLU()),
                           ('fc2', nn.Linear(50, 10))
                           ]))
    def forward(self, x):
        feat = self.features(x)
        x = feat.clone().view(-1, 640)
        x = self.classifier(x)
        logsftm_x = F.log_softmax(x,dim=1)
        return x #feat, x


    
    
    



