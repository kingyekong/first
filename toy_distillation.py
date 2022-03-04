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

# Set Hyperparameters
batch_size = 128
d = 2
m = 2
N_EPOCHS = 10
LEARNING_RATE = 0.001

# Teacher model
class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork,self).__init__()

        self.fc = nn.Linear(d,1)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        teacher = self.active(x)

        return teacher

# Students model
class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()

        self.fc = nn.Linear(d,m)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.active(x)
        students = torch.sum(x)

        return students

torch.manual_seed(RANDOM_SEED)

teacher_model = TeacherNetwork().to(DEVICE)
teacher_model.parameters()
student_model = StudentNetwork().to(DEVICE)
optimizer = optim.SGD(student_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.MSELoss()

# Training Loop

def training_loop(teacher_model, student_model, criterion, optimizer, device, epochs, print_every=1):
    
    student_model.train()
    
    for epoch in range(0, epochs):
        
        optimizer.zero_grad()
        
        X = torch.randn(d,batch_size)
        X = X.to(device)
        
        target = teacher_model(X)
        output = student_model(X)
        
        #Forward Pass
        loss = criterion(output, target)
        
        #Backwardpass
        loss.backward()
        optimizer.step()
           
        return optimizer, loss
        
        if epoch % print_every == (print_every - 1):
            
            print(f'Epoch: {epoch}\t'
                  f'Loss: {loss:.4f}'
                 )
    
optimizer, loss = training_loop(teacher_model, student_model, criterion, optimizer, device, N_EPOCHS)
