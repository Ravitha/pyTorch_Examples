import tqdm
import torch
import pickle
import torchvision
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

#Create Model Class
#Define layers in __init__()
#Define method forward, specifying the sequence of operations

class NN(torch.nn.Module):
	def __init__(self):
		super(NN,self).__init__()
		self.conv1 = torch.nn.Conv2d(1,6,3,padding=1)
		self.pool1 = torch.nn.MaxPool2d(2)
		self.conv2 = torch.nn.Conv2d(6,16,5,padding=0)
		self.pool2 = torch.nn.MaxPool2d(2)
		self.linear1 = torch.nn.Linear(400,120)
		self.linear2 = torch.nn.Linear(120,84)
		self.linear3 = torch.nn.Linear(84,10)
	def forward(self,x):
		c1= F.relu(self.conv1(x))
		s1 = self.pool1(c1)
		c2 = F.relu(self.conv2(s1))
		s2 = self.pool2(c2)
		f  = (torch.nn.Flatten()(s2))
		f1 = F.relu(self.linear1(f))
		f2 = F.relu(self.linear2(f1))
		f3 = self.linear3(f2)
		return f3

def test_data(testLoader, Criterion, device, model):
	model.eval()
	loss = 0
	acc=0
	with torch.no_grad():
		for data,target in testLoader:
			data = data.to(device)
			target = target.to(device)
			y_pred = model(data)
			loss = loss + Criterion(y_pred,target)
			pred_label = y_pred.data.max(1)[1]
			act_label = target
			if(pred_label.item() == act_label.item()):
				acc = acc+1
	return loss,acc


def save_pickle(file, list):
	with open(file,'wb') as f:
		pickle.dump(list,f)

def unpickle_from_file(file):
	with open(file,'rb') as handle:
		return pickle.load(handle)

def train(model, device, Criterion, optimizer, trainLoader, testLoader):
	outer = tqdm.tqdm(total=100, desc='Epoch', position=0)
	model.train()
	epoch_list=[]
	loss_list=[]
	test_loss_list=[]
	test_acc_list=[]
	for epoch in range(100):
		running_loss = 0
		for i,(data,target) in enumerate(trainLoader,0):
			x = data.to(device)
			y = target.to(device)
			#Compute model ouput
			pred = model(x)
			#Compute loss
			loss = Criterion(pred,y)
			running_loss +=loss.item()
			#Optimizer to adjust weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		test_loss, test_acc = test_data(testLoader, Criterion, device, model)
		epoch_list.append(epoch)
		test_loss_list.append(test_loss/len(testLoader))
		loss_list.append(running_loss/len(trainLoader))
		test_acc_list.append(test_acc/10000)
		outer.update(1)
	return epoch_list,loss_list,test_loss_list,test_acc_list


def loaders():
	trainLoader = DataLoader(torchvision.datasets.MNIST('./data',train=True, download=True,
				transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             	])), batch_size=600)
	testLoader = DataLoader(torchvision.datasets.MNIST('./data',train=False, download=True,
				transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             	])), batch_size=1)
	return trainLoader,testLoader

#Create object for model
#Porting model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN().to(device)
#Summarize the model
summary(model,(1,28,28))
#Create Optimizer and Loss Function
Criterion = torch.nn.CrossEntropyLoss(size_average=True) # It computes log softmax and negative log likelihood 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#Get Loaders
trainLoader,testLoader = loaders()
#Train Model
epoch_list, loss_list, test_loss_list, test_acc_list = train(model, device, Criterion, optimizer, trainLoader, testLoader)
#Save the Model state in a file to support inference

save_pickle('TrainLoss.pkl',loss_list)
save_pickle('TestLoss.pkl',test_loss_list)
save_pickle('TestAccuracy.pkl',test_acc_list)

'''
Plot the obtained results
'''
plt.plot()
plt.title('Classification of MNIST digits using LeNet')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epoch_list,loss_list,'go-',label='Train Loss')
plt.plot(epoch_list,test_loss_list,'r*',label='Test Loss')
plt.legend()
plt.show()

'''
Metrics to gauge the Model
'''
print('Test Accuracy:',1-test_acc_list[-1])



