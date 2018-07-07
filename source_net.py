import torch.nn as nn
from functions import ReverseLayerF
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
from data_loader import mnist_loader
from functions import SIMSE, DiffLoss, MSE
from test import test



#####################
# setup optimizer   #
#####################
def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print 'learning rate is set to %f' % current_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


# conv net for mnist classification
class source_net(nn.Module):
    def __init__(self, n_class=10):
        super(source_net, self).__init__()
        ################################
        # shared encoder (dann_mnist)
        ################################
        self.conv = nn.Sequential()
        self.conv.add_module('conv_se1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                                                                  padding=2))
        self.conv.add_module('ac_se1', nn.ReLU(True))
        self.conv.add_module('pool_se1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv.add_module('conv_se2', nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                                                  padding=2))
        self.conv.add_module('ac_se2', nn.ReLU(True))
        self.conv.add_module('pool_se2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential()
        self.fc.add_module('fc_se3', nn.Linear(in_features=7 * 7 * 48, out_features=code_size))
        self.fc.add_module('ac_se3', nn.ReLU(True))

        # classify 10 numbers
        self.pred_class = nn.Sequential()
        self.pred_class.add_module('fc_se4', nn.Linear(in_features=code_size, out_features=100))
        self.pred_class.add_module('relu_se4', nn.ReLU(True))
        self.pred_class.add_module('fc_se5', nn.Linear(in_features=100, out_features=n_class))


    def forward(self, input_data):
        # source private encoder
        res = self.conv(input_data)
        res = res.view(-1, 64 * 7 * 7)
        res = self.fc(res)
        res = pred_class(res)
        return res


######################
# params             #
######################
source_image_root = os.path.join('.', 'dataset', 'mnist')
target_image_root = os.path.join('.', 'dataset', 'mnist_m')
model_root = 'model'
cuda = True
cudnn.benchmark = True
lr = 1e-2
batch_size = 32
image_size = 28
n_epoch = 100
step_decay_weight = 0.95
lr_decay_step = 20000
weight_decay = 1e-6
alpha_weight = 0.01
beta_weight = 0.075
gamma_weight = 0.25
momentum = 0.9


num_works = 1
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# transform for images
img_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#######################
# load data           #
#######################
# load mnist dataset
dataloader_source = torch.utils.data.DataLoader(
    datasets.MNIST(source_image_root, 
        train=True, 
        download=True,
        transform=img_transform
    ),
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_works
)


my_net = source_net(n_class=10)

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
loss_classification = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()

for p in my_net.parameters():
    p.requires_grad = True


TrainLog = open('log.txt', 'w')

#############################
# training network          #
#############################
current_step = 0
for epoch in xrange(n_epoch):
    data_source_iter = iter(dataloader_source)
    i = 0
    while i < len(dataloader_source):
        data_source = data_source_iter.next()
        s_img, s_label = data_source
        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        result = my_net(input_data=input_img)


        source_classification = loss_classification(class_label, result)
        loss += source_classification

       
        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1

    print >> TrainLog, 'source_classification: ' % (source_classification.data.cpu().numpy())

    print >> TrainLog, 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    # torch.save(my_net.state_dict(), model_root + '/dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth')
    test(epoch=epoch, name='mnist', logFile=TrainLog)

print 'done'









