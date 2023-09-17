import torchvision
import torch
from matplotlib import pyplot

from torchvision.io import read_image
from torchvision.models import *
from torch import nn
from collections import OrderedDict

weights = ViT_B_16_Weights.DEFAULT
preprocess = weights.transforms(antialias=True)
print(preprocess)

training_set = torchvision.datasets.CIFAR10("Cifar10/train", transform=preprocess, train=True, download=True)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

print(training_set.data.shape)
#pyplot.imshow(training_set.data[2000])
#pyplot.show()

for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        print(inputs[0].shape)
        
        invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        
        img = invTrans(inputs[0])
        #pyplot.imshow(img.permute(1, 2, 0))
        #pyplot.show()
        break 


#accuracy count
pred = torch.argmax(torch.tensor([[0.6 ,0.3, 0.4],
                                  [0.1, 0.2, 0.3],
                                  [0.2, 0.1, 0.6],
                                  [0.0, 0.1, 0.9]]), dim=1)
print(pred)

labels = torch.tensor([0, 1, 2, 2])


def accuracy_score(correct, total):
    return correct/total

print(accuracy_score((pred==labels).count_nonzero().item(), 4))
