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

mse_list = []
w_list = []
for w in np.arange(0.0,4.1,0.1):
	l=0;
	for x,y in zip(x_data,y_data):
		l = l+loss(x,y)
	mse_list.append(l/3)
	w_list.append(w)

print(mse_list)
print(w_list)

plt.plot(w_list,mse_list)
plt.show()
