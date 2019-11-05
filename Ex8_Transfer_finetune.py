
"""
Author : Ravitha

"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

def train(model, device, criterion, optimizer, dataLoader, testLoader, scheduler):
  best_acc = 0
  best_model_wts = copy.deepcopy(model.state_dict())
  for epoch in range(25):
     # set the model in training mode
    model.train()
    run_loss = 0.0
    run_acc = 0.0
    size = 0
    for inputs,labels in dataLoader:
      inputs = inputs.to(device)
      labels = labels.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()

      # Find the label for the input
      # Compute Loss and gradients
      # update weights
      with torch.set_grad_enabled(True):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        size += inputs.size(0)
        run_loss += loss.item() * inputs.size(0)
        run_acc += torch.sum(preds == labels.data)
    scheduler.step()
    epoch_loss = run_loss / size
    epoch_acc = run_acc.double() / size
    valid_loss, valid_acc = valid(model, device, criterion, optimizer, dataLoader, testLoader)
    print('Epoch: {} Train Loss: {:.2f} Train Acc: {:.2f} Valid Loss: {:.2f} Valid Acc: {:.2f}'
           .format(epoch,epoch_loss,epoch_acc,valid_loss,valid_acc))
    if(valid_acc>best_acc):
      best_model_wts = copy.deepcopy(model.state_dict())
  return best_model_wts

def valid(model, device, criterion, optimizer, dataLoader, testLoader):
  # set the model in training mode
  model.eval()
  run_loss = 0.0
  run_acc = 0.0
  size = 0
  for inputs,labels in testLoader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # zero the parameter gradients
    optimizer.zero_grad()

    # Find the label for the input
    # Compute Loss and gradients
    # update weights
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)
      run_loss += loss.item() * inputs.size(0)
      run_acc += torch.sum(preds == labels.data)
      size += inputs.size(0)
  epoch_loss = run_loss / size
  epoch_acc = run_acc.double()/size
  return epoch_loss, epoch_acc

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

weights = train(model_ft, device, criterion, optimizer_ft, dataloaders['train'], dataloaders['val'], exp_lr_scheduler)
torch.save({'model_state_dict':model_ft.state_dict(),
		'optim_state_dict':optimizer_ft.state_dict()},'resnet18_classification_weights.pth')

