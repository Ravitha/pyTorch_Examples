'''
Utilizing a pretrained models and make inference on new images

'''
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt

with open("/home/ravitha/D/pyTorch_models/class.json", "r") as read_file:
    class_idx = json.load(read_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# it creates model with random weights
alexnet = models.alexnet(pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocessing = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

image_path = '/home/ravitha/D/pyTorch_models/Images/dog.jpeg'
image_sample =  Image.open(image_path)
transformed_sample = preprocessing(image_sample)
transformed_numpy = transformed_sample.numpy()
#print(transformed_numpy.shape)

transformed_numpy=np.moveaxis(transformed_numpy,0,-1)
#print(transformed_numpy.shape)
# plot the input image
plt.imshow(transformed_numpy)
plt.show()

#print(transformed_sample.size())
transformed_sample = transformed_sample.view([1,3,224,224])
#print(transformed_sample.size())
alexnet.eval()

predict = alexnet(transformed_sample)
#print(predict)

idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
pred_label = predict.data.max(1)[1]
print(idx2label[pred_label])
