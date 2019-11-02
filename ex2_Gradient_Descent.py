import numpy as np
from matplotlib import pyplot as plt

x_data =[1.0,2.0,3.0]
y_data =[2.0,3.0,4.0]
w = 1.0

def forward(x):
	return w*x

def loss(x,y):
	y_pred = forward(x)
	return (y_pred-y) * (y_pred - y)

def gradient(x,y,w):
	return 2*x*(w*x-y)

epochs=[]
loss_epoch=[]
for epoch in range(100):
	for(x,y) in zip(x_data,y_data):
		grad = gradient(x,y,w)
		w =  w - grad*0.01
		l = loss(x,y)
	print(str(epoch)+":"+str(l))
	epochs.append(epoch)
	loss_epoch.append(l)



plt.plot(epochs,loss_epoch)
plt.show()
