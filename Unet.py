from torch.autograd import Variable
from skimage.transform import resize
from collections import defaultdict
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import os
import torch
import numpy as np
import  scipy.misc as smp
import copy
import time
from scipy.ndimage import imread
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
'''
Images in Nuclei_dataset are of dimension 512 * 512 * 4 
Each dimension has a grayscale intendity between 0 and 255
Masks are of dimension 512 * 512 
Nuclei has pixel intensity of 255 and background has an intensity zero
'''
num_Epochs = 100
def trainData():
	data_location = './Nuclei_Dataset/Image'
	mask_location = './Nuclei_Dataset/Mask'
	train_data = []
	train_data_gt = []
	for img in os.listdir(data_location):
		train_data.append(os.path.join(data_location,img))
		train_data_gt.append(os.path.join(mask_location,img))

	numImages = len(train_data)
	train_Images = np.zeros(shape=(numImages,224,224,3))
	train_labels = np.zeros(shape=(numImages,224,224,1))

	for file_index in range(numImages):
		image = imread(train_data[file_index])
		if(image.shape[-1] == 4):
			image = image[:,:,0:3]
		train_Images[file_index,:,:] = resize(image,(224,224,3))
		label = imread(train_data_gt[file_index]).reshape(512,512,1)
		train_labels[file_index,:,:] = resize(label,(224,224,1))
	train_Images.astype('float32')
	train_labels.astype('float32')
	train_Images = train_Images/255.0
	train_mean = [train_Images[...,i].mean() for i in range(train_Images.shape[-1])]
	train_std = [train_Images[...,i].std() for i in range(train_Images.shape[-1])]


	train_Images = (train_Images - train_mean)/train_std
	train_labels = train_labels/255.0
	
	train_Images = np.transpose(train_Images,(0,3,1,2))
	train_labels = np.transpose(train_labels,(0,3,1,2))
	return train_Images, train_labels, train_mean, train_std


def testData(train_mean, train_std):
	data_location = './Nuclei_Dataset/Test_Image'
	mask_location = './Nuclei_Dataset/Test_Masks'
	test_data = []
	test_data_gt = []
	for img in os.listdir(data_location):
		test_data.append(os.path.join(data_location,img))
		test_data_gt.append(os.path.join(mask_location,img))

	numImages = len(test_data)
	test_Images = np.zeros(shape=(numImages,224,224,3))
	test_labels = np.zeros(shape=(numImages,224,224,1))

	for file_index in range(numImages):
		image = imread(test_data[file_index])
		if(image.shape[-1] == 4):
			image = image[:,:,0:3]
		test_Images[file_index,:,:] = resize(image,(224,224,3))
		label = imread(test_data_gt[file_index]).reshape(512,512,1)
		test_labels[file_index,:,:] = resize(label, (224,224,1))

	test_Images.astype('float32')
	test_labels.astype('float32')
	test_Images = test_Images/255.0
	
	test_Images = (test_Images - train_mean)/train_std
	test_labels = test_labels/255.0
	test_Images = np.transpose(test_Images,(0,3,1,2))
	test_labels = np.transpose(test_labels,(0,3,1,2))
	return test_Images, test_labels

class SegDataset(Dataset):
	def __init__(self, images, labels):
		self.image =  torch.from_numpy(images)
		self.gt = torch.from_numpy(labels)

	def __len__(self):
		return self.image.shape[0]

	def __getitem__(self,index):
		return self.image[index], self.gt[index]


class FCN(torch.nn.Module):
	def __init__(self, model, n_class):
		super().__init__()
		layers = list(model.children())
		self.layer1 = nn.Sequential(*layers[:5])
		self.upsample1 = nn.Upsample(scale_factor = 4, mode='bilinear')
		self.layer2 = layers[5]
		self.upsample2 = nn.Upsample(scale_factor = 8, mode='bilinear')
		self.layer3 = layers[6]
		self.upsample3 = nn.Upsample(scale_factor = 16, mode='bilinear')
		self.layer4 = layers[7]
		self.upsample4 = nn.Upsample(scale_factor = 32, mode='bilinear')

		self.conv1k = nn.Conv2d(64+128+256+512, n_class, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = self.layer1(x)
		up1 = self.upsample1(x)
		x = self.layer2(x)
		up2 = self.upsample2(x)
		x = self.layer3(x)
		up3 = self.upsample3(x)
		x = self.layer4(x)
		up4 = self.upsample4(x)
		merge = torch.cat([up1,up2,up3,up4], dim=1)
		merge = self.conv1k(merge)
		#out = self.sigmoid(merge)
		return merge

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return out

def dice_loss(pred, target, smooth = 1.):
	pred = pred.contiguous()
	target = target.contiguous()
	intersection = torch.sum(pred * target)
	loss = (1 - ((2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)))
	return loss.mean()


def calc_loss(pred, target, metrics, Criterion, bce_weight=0.5):
	#bce = Criterion(pred, target)
	#bce = F.binary_cross_entropy_with_logits(pred, target)
	pred = torch.sigmoid(pred)
	bce = Criterion(pred, target)
	#target =target.type(torch.cuda.LongTensor)
	#bce = F.binary_cross_entropy_with_logits(pred,target)
	dice = dice_loss(pred, target)
	loss = bce * bce_weight  
	metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
	#metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

	return loss

def print_metrics(metrics, epoch_samples, phase):    
	outputs = []
	for k in metrics.keys():
		outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
	print("{}: {}".format(phase, ", ".join(outputs)))   

def validation(model, optimizer, testLoader, device, Criterion):
	model.eval()
	metrics = defaultdict(float)
	size =0
	vloss =0 
	with torch.no_grad():
		for vinput,vlabel in testLoader:
			vinput = Variable(vinput)
			vinput = vinput.to(device)
			vinput = vinput.type(torch.cuda.FloatTensor)
			vlabel = Variable(vlabel)
			vlabel = vlabel.to(device)
			vlabel = vlabel.type(torch.cuda.FloatTensor)
			optimizer.zero_grad()
			vpredict = model(vinput)
			vloss = calc_loss(vpredict, vlabel, metrics, Criterion)
			size = size + vinput.size(0)
	print_metrics(metrics, size, 'val')
	epoch_loss = metrics['loss']/ size
	return epoch_loss

def train_model(model, optimizer, dataLoader, testLoader, device, Criterion, scheduler):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 1e10
	for epoch in range(num_Epochs):
		model.train()
		metrics = defaultdict(float)		
		size = 0
		for image,target in dataLoader:
			image = Variable(image)
			image = image.to(device)
			image = image.type(torch.cuda.FloatTensor)
			target = Variable(target)
			target = target.to(device)
			target = target.type(torch.cuda.FloatTensor)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				predict = model(image)
				loss = calc_loss(predict, target, metrics, Criterion)
				loss.backward()
				optimizer.step()
				scheduler.step()
				size += image.size(0)
		print_metrics(metrics, size, 'train')
		test_loss = validation(model, optimizer, testLoader, device, Criterion)
		if test_loss<best_loss:
			best_loss = test_loss
			best_model_wts = copy.deepcopy(model.state_dict())
	model.load_state_dict(best_model_wts)
	return model

# Choose Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create DataLoaders
train_image, train_mask, train_mean, train_std = trainData()
dataset = SegDataset(train_image, train_mask)
train = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

test_image, test_mask = testData(train_mean, train_std)
dataset = SegDataset(test_image, test_mask)
test = DataLoader(dataset=dataset, batch_size=5, shuffle=True)

# Create Neural Network Model
#base_model= models.resnet18(pretrained = False)
fcn_model = UNet(1).to(device)
Criterion = nn.BCELoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(fcn_model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1)
model = train_model(fcn_model, optimizer, train, test, device, Criterion, exp_lr_scheduler)

print('Predictions')
inputs, targets = next(iter(train))
inputs = inputs.to(device)
inputs = inputs.type(torch.cuda.FloatTensor)
targets = targets.to(device)
targets = targets.type(torch.cuda.FloatTensor)
fcn_model.eval()
prediction = fcn_model(inputs)

inputs = inputs.cpu()
targets = targets.cpu()
prediction = prediction.cpu()
prediction = torch.sigmoid(prediction)

image = inputs[0].numpy()

#plt.subplot(1,3,1)
#plt.imshow(image)


image = image/255.0
for i in range(0,3):
	image[:,:,i] = (image[:,:,i]-train_mean[i])/train_std[i]
image = image.transpose((1,2,0))

target = targets[0].numpy().transpose((1,2,0))
target = target.reshape(224,224)
prediction = Variable(prediction)
pred = prediction[0].numpy().transpose((1,2,0))
pred = pred.reshape(224,224)
#pred = np.argmax(pred, 1-pred)




plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(target)
plt.subplot(1,3,3)
plt.imshow(pred)
plt.show()
