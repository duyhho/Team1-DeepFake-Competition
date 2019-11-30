
# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

data_dir = './data/face_data'
# data_dir = './classifiers/inferences/'

batchSize = 20
imageSize = 64
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
#
# dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 0) # We use dataLoader to get the images of the training set batch by batch.
#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output
def predict_image(image,num):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
    output = model(noise)
    fake = model(noise)

    vutils.save_image(fake.data, '%s/fake_%03d.png' % ("./fake",num), normalize=True)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=transform)
    global classes
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    # dataset = dset.CIFAR10(root='./data', download=True,
    #                        transform=transform)  # We download the training set in the ./data folder and we apply the previous transformations on each image.
    # loader = torch.utils.data.DataLoader(dataset, batch_size = num, shuffle = True, num_workers = 0) # We use dataLoader to get the images of the training set batch by batch.
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels
model = G()
model.load_state_dict(torch.load('./results/generator-024.pth'))
model.eval()
classes = []

to_pil = transforms.ToPILImage()
images, labels = get_random_images(100)
for ii in range(len(images)):
    # print(images)
    vutils.save_image(images[ii], '%s/real_%03d.png' % ("./real", ii) , normalize=True)
    image = to_pil(images[ii])
    index = predict_image(image,ii)

# dataset = dset.CIFAR10(root='./data', download=True,
#                            transform=transform)  # We download the training set in the ./data folder and we apply the previous transformations on each image.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True, num_workers = 0) # We use dataLoader to get the images of the training set batch by batch.

# data = datasets.ImageFolder(data_dir, transform=transform)
# classes = data.classes
# indices = list(range(len(data)))
# # np.random.shuffle(indices)
# idx = indices[:batchSize]
# from torch.utils.data.sampler import SubsetRandomSampler
# sampler = SubsetRandomSampler(idx)
# print(sampler)
# dataloader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=batchSize)
#
# for i, data in enumerate(dataloader, 0):
#     real, _ = data
#     input = Variable(real)
#     noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
#     fake = model(noise)
#     vutils.save_image(real, '%s/real_samples.png' % ("./fake_inferences"), normalize=True)
#     vutils.save_image(fake.data, '%s/fake_inference.png' % ("./fake_inferences"), normalize=True)
# #
