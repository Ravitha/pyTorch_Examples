import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torchvision import models,transforms
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt

def preprocess_Image(Image_path):
	normalize = transforms.Normalize(
			mean = [0.485, 0.456, 0.406],
			std = [0.229, 0.224, 0.225]
			)
	preprocess = transforms.Compose([
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			normalize
			])
	Img =  Image.open(Image_path)
	#plt.imshow(Img)
	#plt.show()
	Img_tensor = preprocess(Img)
	Img_Variable = Variable(Img_tensor.unsqueeze(0))
	return Img, Img_Variable


def retrieve_features_classes(preprocessed_image, layer):
	model = models.resnet18(pretrained = True)
	model.eval()

	features_blob = []
	def hook_feature(module, input, output):
		features_blob.append(output.data.cpu().numpy())

	model._modules.get(layer).register_forward_hook(hook_feature)

	logit = model(preprocessed_image)
	class_prob = F.softmax(logit, dim=1).data.squeeze()
	probs, idx = class_prob.sort(0,True)
	probs = probs.numpy()
	idx = idx.numpy()

	params = list(model.parameters())
	weights = np.squeeze(params[-2].data.numpy())


	return features_blob[0], probs, idx, weights


def generate_cam(features, weights, class_id):
	bs, nc, h, w = features.shape
	cam = weights[class_id].dot(features.reshape((nc, h*w)))
	cam = cam.reshape(h,w)
	cam = cam - np.min(cam)
	cam = cam / np.max(cam)
	cam_img = np.uint8(255 * cam)
	cam_img = cv2.resize(cam_img, (224,224))
	return cam_img



Image_path ='./Images/dog_cat.jpeg'
img, Tensor = preprocess_Image(Image_path)
feature, probs, idx, weights = retrieve_features_classes(Tensor, 'layer4')
print('Majority class : {0}',idx[0])
cam_img = generate_cam(feature, weights, idx[0])
plt.imshow(img)
plt.imshow(cam_img, cmap='jet', alpha = 0.5)
plt.show()
