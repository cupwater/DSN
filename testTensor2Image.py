import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader, mnist_m_loader
from functions import SIMSE, DiffLoss, MSE
from test import test


mnist_m_data = torch.load('./dataset/mnist_m/processed/mnist_m_train.pt')

img = mnist_m_data[0][0]
img = img.numpy()

from PIL import Image


img = Image.fromarray(img, 'RGB')