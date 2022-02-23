import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
import torchvision.models as models

import time
import os
import copy
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt


#Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#Data setting
x = torch.randn(9,4)
d = x.ndim

#Parameters
LEARNING_RATE = 0.001

#Data
m = 2
w0 = 2

train_dataset = datasets.x(train=True)
valid_dataset = datasets.x(train=False)

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False)

#Teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel,self).__init__()

        self.fc = F.Linear(x,w0)

    def forward(self, x):
        x = x.fc(x)
        teacher = F.relu(x)

        return teacher

#Students model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.fc = nn.Linear(d,m)

    def forward(self, x):
        x = x.fc(x)
        x = F.relu(x)
        students = torch.sum(x)

        return students

optimizer = optim.SGD(StudentModel.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer.zero_grad()
loss = F.mse_loss(teacher, students, 2)
