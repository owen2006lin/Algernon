import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),                      # PIL Image -> FloatTensor [C,H,W] in [0,1]
    transforms.Normalize((0.5,), (0.5,))        # optional; use (0.5,0.5,0.5) for RGB like CIFAR-10
])


train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
val_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)

# Even |-> true, Odd |-> false
train_data.targets = torch.where(train_data.targets % 2 == 0, 1 , 0)
val_data.targets = torch.where(val_data.targets % 2 == 0, 1 , 0)
train_data.classes = ['odd', 'even']
val_data.classes = ['odd', 'even']

training_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
validation_loader = DataLoader(val_data, batch_size = 128, shuffle = False)

# Plotting
'''
import matplotlib.pyplot as plt
import numpy as np
def matplotlib_imshow(img, one_channel = False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
classes = train_data.classes
print('  '.join(classes[labels[j]] for j in range(4)))
plt.show()
'''

import torch.nn as nn

class ParityMLP():
    def __init__(self):


