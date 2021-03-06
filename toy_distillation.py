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


# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data setting
d = x.ndim
m = 2
w0 = 2
x = torch.randn(2,400)

# Parameters
LEARNING_RATE = 0.001
N_CLASSES = 10

train_dataset = datasets.x(train=True)
valid_dataset = datasets.x(train=False)

train_loader = DataLoader(dataset=train_dataset, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, shuffle=False)

# Teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel,self).__init__()

        self.fc = F.Linear(x,w0)
        self.active = F.relu()

    def forward(self, x):
        x = self.fc(x)
        teacher = self.active(x)

        return teacher

# Students model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.fc = nn.Linear(d,m)
        self.active = F.relu()

    def forward(self, x):
        x = self.fc(x)
        x = self.active(x)
        students = torch.sum(x)

        return students

torch.manual_seed(RANDOM_SEED)

model = StudentModel(N_CLASSES).to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = F.mse_loss()

# Training

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    
    for epoch in range(0, epochs):
        model.train()
        running_loss = 0
        
        for x in train_loader:
            
            optimizer.zero_grad()
            
            x = x.to(device)
            teacher = teacher.to(device)
            students = students.to(device)
            
            #Forward Pass
            loss = criterion(students, teacher)
            
            #Backwardpass
            loss.backward()
            optimizer.step()

