import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

#Create Model Class
#Define layers in __init__()
#Define method forward, specifying the sequence of operations

class NN(torch.nn.Module):
	def __init__(self):
		super(NN,self).__init__()
		self.linear = torch.nn.Linear(1,1)

	def forward(self,x):
		y_pred = F.sigmoid(self.linear(x))
		return y_pred

#Create object for model
model = NN()

#Create Optimizer and Loss Function

Criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Train

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data = Variable(torch.Tensor([[1.],[0.],[1.]]))

epoch_list=[]
loss_list = []

for epoch in range(100):
	#Compute model ouput
	pred = model(x_data)

	#Compute loss
	loss = Criterion(pred,y_data)

	#Optimizer to adjust weights
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	epoch_list.append(epoch)
	loss_list.append(loss)

plt.plot(epoch_list,loss_list)
plt.show()

#Test

x_test = torch.Tensor([[1.0]])
print(model(x_test))
