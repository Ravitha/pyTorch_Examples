'''
Fine Tune the Conventional CNN for any task at hand
Author : Ravitha N
Refer to the link https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html for more information
1. ResNet18 model is created
2. The model parameters are initialized using pretrained weights
3. Last fully connected layer is modified to contain two nodes (two classes)
4. The modified model is fine tuned for new application using SGD
5. The model state which provides better validation accuracy is tracked.
6. Finally, the model state is stored in '.pth' file
'''
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models,datasets,transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import copy

'''
Set Default Parameters
'''
input_size = 224
batch_size = 4
num_Epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(model, Criterion, optimizer, testLoader):
	model.eval()
	running_loss = 0.0
	acc =0.0
	size =0
	for vinput,vlabel in testLoader:
		vinput = vinput.to(device)
		vlabel = vlabel.to(device)
		optimizer.zero_grad()
		with torch.set_grad_enabled(False):
			vpredict = model(vinput)
			vloss = Criterion(vpredict, vlabel)
			_, vpreds = torch.max(vpredict,1)
			size = size + vinput.size(0)
			running_loss += vloss.item() * vinput.size(0)
			acc += torch.sum(vpreds == vlabel.data)
	return running_loss/size, acc.double()/size

def train_model(model, Criterion, optimizer, dataLoader, testLoader):
	for epoch in range(num_Epochs):
		model.train()
		running_loss =0.0
		acc = 0.0
		size = 0
		for input,label in dataLoader:
			input = input.to(device)
			label = label.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				predict = model(input)
				loss = Criterion(predict, label)
				_, preds = torch.max(predict,1)
				loss.backward()
				optimizer.step()
				size += input.size(0)
				running_loss += loss.item() * input.size(0)
				acc += torch.sum(preds == label.data)
		test_loss, test_acc = validation(model, Criterion, optimizer, testLoader)
		print('Epoch: {} Train Loss: {:.2f} Train Acc: {:.2f} Valid Loss: {:.2f} Valid Acc: {:.2f}'.format(
			epoch, running_loss/size, acc.double()/size, test_loss, test_acc))


def main():

	#Instantiating ResNet model with pretrained weights
	model = models.resnet18(pretrained = True)
	num_features = model.fc.in_features
	#Modifying the fully connected layer to reduce number of classes from 1000 to 2
	model.fc = nn.Linear(num_features, 2)
	model = model.to(device)
	'''
	Fine tuning the network adjust all the parameters in the network 
	optimizer accepts the list of parameters to be updataed
	'''
	params_to_update = model.parameters()
	optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)
	Criterion = nn.CrossEntropyLoss()
	'''
	Creating a DataLoader for Training and Validation 
	'''
	data_dir = 'hymenoptera_data'
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
	train_dataset = datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train'])
	valid_dataset = datasets.ImageFolder(os.path.join(data_dir,'val'),data_transforms['val'])

	dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
	testloader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True)
	weights = train_model(model, Criterion, optimizer, dataloader, testloader)
	torch.save({'model_state_dict':model.state_dict(),
                'optim_state_dict':optimizer.state_dict()},'resnet18_classification_weights.pth')

if __name__ == "__main__":
    main()


