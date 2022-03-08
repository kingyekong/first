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
    
def plot_losses(train_losses):
    
    plt.style.use('seaborn')
    
    train_losses = np.array(train_losses)
    
    fig, ax = plt.subplots(figsize = (8,405))
    
    ax.plot(train_losses, color='red', label='Training loss')
    ax.set(title = "Loss over epochs", xlabel='Epoch', ylabel='Loss')
    ax.legend()
    fig.show()
    
    plt.style.use('default')
    
def train(train_loader, teacher_model, student_model, crieterion, optimizer, device):
    
    for X in train_loader:
        
        optimizer.zero_grad()
        
        X = X.to(device)
                
        target = teacher_model(X)
        output = student_model(X)
        
        #Forward pass
        loss = criterion(output, target)
        
        #Backward pass
        loss.backward()
        optimizer.step()
     
    return student_model, teacher_model, optimizer, epoch_loss

# Training Loop

def training_loop(teacher_model, student_model, criterion, optimizer, device, epochs, print_every=1):
    
    train_losses = []
    student_model.train()
    
    for epoch in range(0, epochs):
        
        #training
        optimizer, train_loss = train(train_loader, teacher_model, student_model, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        if epoch % print_every == (print_every - 1):
            
            print(f'Epoch: {epoch}\t'
                  f'Loss: {train_loss:.4f}'
                 )
 plot_losses(train_losses)

return student_model, teacher_model, optimizer, train_losses

train_loader = torch.randn(d,batch_size)
            
torch.manual_seed(RANDOM_SEED)

teacher_model = TeacherNetwork().to(DEVICE)
teacher_model.parameters()
student_model = StudentNetwork().to(DEVICE)
optimizer = optim.SGD(student_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
criterion = nn.MSELoss()            
    
optimizer, loss = training_loop(teacher_model, student_model, criterion, optimizer, device, N_EPOCHS)
