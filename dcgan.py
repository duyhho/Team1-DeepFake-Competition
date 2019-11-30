
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
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

data_dir = './data/face_data'
batchSize = 64 
imageSize = 64 

transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

dataset = dset.ImageFolder(data_dir, transform=transform)
# dataset = dset.CIFAR10(root = './data', download = True, transform = transform) # We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 0) # We use dataLoader to get the images of the training set batch by batch.

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

netG = G()
netG.apply(weights_init)
netG.cuda()

class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

netD = D()
netD.apply(weights_init)
netD.cuda()

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

loss_D_min = float('inf')
for epoch in range(25):

    for i, data in enumerate(dataloader, 0):
        # As training progresses the discriminator improves at this task.
        # But our end goal is attained at a theoretical point where the discriminator outputs 0.5
        # for both types of inputs (i.e.indecisive if fake or real).
        netD.zero_grad()
        
        real, _ = data
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(input.cuda())
        errD_real = criterion(output, target)
        output = (output > 0.5).float()
        correct_D_fake = ((output == target).float().sum())/output.shape[0]

        noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
        fake = netG(noise.cuda())
        target = Variable(torch.zeros(input.size()[0])).cuda()
        output = netD(fake.detach())
        errD_fake = criterion(output, target)
        output = (output > 0.5).float()
        correct_D_real = ((output == target).float().sum())/output.shape[0]

        errD = errD_real + errD_fake
        correct_D = (correct_D_real + correct_D_fake)/2
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0])).cuda()
        output = netD(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizerG.step()
        output = (output > 0.5).float()
        # print(output)
        # print(target)
        correct_G = ((output == target).float().sum()) / output.shape[0]

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D_Accuracy: %.3f G_Accuracy: %.3f' % (epoch, 25, i, len(dataloader), errD.data, errG.data, correct_D, correct_G))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples.png' % "./results", normalize = True)
            fake = netG(noise.cuda())
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
            torch.save(netG.state_dict(), '%s/generator-%03d.pth' % ("./results", epoch))
            # # Accuracy
            # output = (output > 0.5).float()
            #
            # print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, 25, errD.data,
            #                                                            correct_D))
            # print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, 25, errD.data,
            #                                                            correct_D))

