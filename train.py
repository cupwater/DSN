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
 # only encoding target and source domain dataset, no adversarial. 
 # when finished encoding, then adversarial traning to align source and target domain
active_domain_loss_step = 10000
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

# # load mnist_m dataset
# mnist_data = torch.load('./dataset/mnist/processed/training.pt')
# dataloader_source = torch.utils.data.DataLoader(
#     mnist_loader(
#         data = mnist_data[0], 
#         label = mnist_data[1],
#         train=True,
#         transform=img_transform
#     ),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_works
# )


#load mnist_m dataset
mnist_m_data = torch.load('./dataset/mnist_m/processed/mnist_m_train.pt')
dataloader_target = torch.utils.data.DataLoader(
    mnist_loader(
        data = mnist_m_data[0], 
        label = mnist_m_data[1],
        train=True,
        transform=img_transform
    ),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_works
)

#####################
#  load model       #
#####################
my_net = DSN()


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


optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.CrossEntropyLoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True



TrainLog = open('log.txt', 'w')

#############################
# training network          #
#############################
len_dataloader = min(len(dataloader_source), len(dataloader_target))
print len(dataloader_source)
print len(dataloader_target)
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

current_step = 0
for epoch in xrange(n_epoch):
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    # print 'debug 1'

    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = data_target_iter.next()
        t_img, t_label = data_target

        my_net.zero_grad()
        loss = 0
        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        # print 'debug 2'

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        target_inputv_img = Variable(input_img)
        target_classv_label = Variable(class_label)
        target_domainv_label = Variable(domain_label)

        if current_step > active_domain_loss_step:
            p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
            p = 2. / (1. + np.exp(-10 * p)) - 1
            # activate domain loss
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
            target_privte_code, target_share_code, target_domain_label, target_class_label, target_rec_code = result
            # target_privte_code, target_share_code, target_domain_label, target_rec_code = result
            target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
            loss += target_dann
        else:
            target_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
            # target_privte_code, target_share_code, _, target_rec_code = result
            target_privte_code, target_share_code, _, target_class_label, target_rec_code = result

        # print 'debug 3'
        # classification loss for target domain
        target_classification = loss_classification(target_class_label, target_classv_label)
        loss += target_classification
        # different loss
        target_diff= beta_weight * loss_diff(target_privte_code, target_share_code)
        loss += target_diff
        
        # reconstruct, L1 + L2 reconstruct loss
        target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
        loss += target_mse
        target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
        loss += target_simse
        # target_mse = 0
        # target_simse = 0


        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################

        data_source = data_source_iter.next()
        s_img, s_label = data_source

        # print 'debug 4'

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        loss = 0

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(input_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        source_inputv_img = Variable(input_img)
        source_classv_label = Variable(class_label)
        source_domainv_label = Variable(domain_label)

        if current_step > active_domain_loss_step:
            # activate domain loss
            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
            source_privte_code, source_share_code, source_domain_label, source_class_label, source_rec_code = result
            source_dann = gamma_weight * loss_similarity(source_domain_label, source_domainv_label)
            loss += source_dann
        else:
            source_dann = Variable(torch.zeros(1).float().cuda())
            result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
            source_privte_code, source_share_code, _, source_class_label, source_rec_code = result

        source_classification = loss_classification(source_class_label, source_classv_label)
        loss += source_classification

        # print 'debug 5'
        # different loss
        source_diff = beta_weight * loss_diff(source_privte_code, source_share_code)
        loss += source_diff

        # reconstruct loss
        source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
        loss += source_mse
        source_simse = alpha_weight * loss_recon2(source_rec_code, source_inputv_img)
        loss += source_simse
        
        # source_mse = 0
        # source_simse = 0

        loss.backward()
        optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
        optimizer.step()

        i += 1
        current_step += 1
    # print >> TrainLog, 'source_classification: %f, source_dann: %f, source_diff: %f, ' \
    #       'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
    #       'target_mse: %f, target_simse: %f' \
    #       % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
    #          source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
    #          target_diff.data.cpu().numpy(),target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy())

    print >> TrainLog, 'source_classification: %f, source_dann: %f, source_diff: %f, ' \
          'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, ' \
          'target_mse: %f, target_simse: %f' \
          % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(), source_diff.data.cpu().numpy(),
             source_mse, source_simse, target_dann.data.cpu().numpy(),
             target_diff.data.cpu().numpy(),target_mse, target_simse)

    print >> TrainLog, 'step: %d, loss: %f' % (current_step, loss.cpu().data.numpy())
    torch.save(my_net.state_dict(), model_root + '/dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth')
    print 'debug 1'
    test(epoch=epoch, name='mnist', logFile=TrainLog)
    print 'debug 2'
    test(epoch=epoch, name='mnist_m',  logFile=TrainLog)
    print 'debug 3'

print 'done'





