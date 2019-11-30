# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

data_dir = './classifiers/inferences'
data_dir = './video-to-frames/data-resized'

test_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transforms2 = transforms.Compose(
                                      [transforms.ToTensor()],
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     )
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms2)
    global classes
    classes = ['fake', 'real']
    classes = data.classes
    print(classes)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
model.cuda()
model.load_state_dict(torch.load('./classifiers/classifier.pth'))
model.eval()

total_num = 30
rows = 3
res = 0
to_pil = transforms.ToPILImage()
images, labels = get_random_images(total_num)
# print(images)
print(labels)
fig=plt.figure(figsize=(15,15))
fig.set_tight_layout(True)

# x = range(total_num)
# y = range(total_num)
#
# fig, ax = plt.subplots(nrows=2, ncols=10)
#
# for row in ax:
#     for col in row:
#         col.plot(x, y)
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    print(index)
    print(classes)
    sub = fig.add_subplot(rows, len(images)/rows, ii+1) #row by len/row grid
    title = 'fake'
    if index != 0:
        title = 'real'
    res += index
    sub.set_title('L:%s\nP:%s' % (classes[labels[ii]], title))
    plt.axis('off')
    if index == 0:
        plt.axis('on')
        plt.setp(sub.spines.values(), color="red")
    imgplot = plt.imshow(image)

fig.suptitle('Deepfake Detection')
plt.text(0, 100, "Fake Ratio: {:.2%}".format(1 - res/total_num), fontsize = 13,color = 'yellow', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
plt.show()