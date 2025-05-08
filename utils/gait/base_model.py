# OpenGait base model for inference
# Copied from OpenGait/opengait/modeling/base_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import os
import sys
import time
import math
import random
import logging
from typing import List, Dict, Tuple, Union

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(__name__)

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        self.logger.info(f"Model loaded from {path}")

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def print_network(self):
        num_params = sum(p.numel() for p in self.parameters())
        self.logger.info(f"Network architecture:\n{self}\nTotal number of parameters: {num_params}")