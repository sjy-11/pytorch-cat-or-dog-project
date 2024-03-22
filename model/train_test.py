from model import Net

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import PIL as Image

#initalise the model
net = Net()

#prepare the train and test datasets
