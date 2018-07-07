import multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
	mp.set_start_method('spawn')
import argparse
import os
import sys
import time
import yaml
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from tensorboardX import SummaryWriter
import logging

import models
import models.resnet
#from models.MobileNetV2 import mobilenet_v2		#no bn-sync
from utils import AverageMeter, accuracy, load_state, save_state, create_logger, IterLRScheduler, param_groups
from multitask_mimic_dataset import MultitaskMimicDataset
from distributed_utils import dist_init, average_gradients, DistModule, DistributedGivenSizeSampler, DistributedGivenIterationSampler, simple_group_split
from torchE.nn import SyncBatchNorm2d

from losses import L2Loss
from prunedutils._save_helper import save_model_def
from lib.mobilenet.MobileNetV2 import mobilenet_v2


model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Multi-Task Training')
parser.add_argument('--last-iter', default=-1, type=int)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--ignore', nargs='+', default=[], type=str)
parser.add_argument('--recover', dest='recover', action='store_true')
parser.add_argument('--load-single', dest='load_single', action='store_true')
parser.add_argument('--port', default='23500', type=str)
parser.add_argument('--config', default='', type=str)
parser.add_argument('--save-path', default='', type=str)


global train_state
train_state = True


#broadcast all base layer to all node, broadcast logit to the specific task
class MultiTaskMimic(nn.Module):
	def __init__(self, feature_dim, num_classes, fc_std=0.001):
		super(MultiTaskMimic, self).__init__()
		self.logits = nn.Linear(feature_dim, num_classes)
		self.logits.weight.data.normal_(std=fc_std)
	def forward(self, x):
		x = self.logits(x)
		return x


#decoder, reconstruct face images
class Generator():
    def __init__(self, input_dim, output_size=(56,56)):
        self.super(Generator)

        self.shared_decoder_fc = nn.Sequential()
        self.shared_decoder_fc.add_module('fc_sd1', nn.Linear(in_features=code_size, out_features=588))
        self.shared_decoder_fc.add_module('relu_sd1', nn.ReLU(True))

        self.shared_decoder_conv = nn.Sequential()
        self.shared_decoder_conv.add_module('conv_sd2', nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd2', nn.ReLU())

        self.shared_decoder_conv.add_module('conv_sd3', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5,
                                                                  padding=2))
        self.shared_decoder_conv.add_module('relu_sd3', nn.ReLU())

        self.shared_decoder_conv.add_module('us_sd4', nn.Upsample(scale_factor=2))

        self.shared_decoder_conv.add_module('conv_sd5', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                                                                  padding=1))
        self.shared_decoder_conv.add_module('relu_sd5', nn.ReLU(True))

        self.shared_decoder_conv.add_module('conv_sd6', nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3,
                                                                  padding=1))


# contain domain_encoder, shared_encoder, shared_decoder, domain_classifier, task_classifier
class DSN(nn.Module):

    # domain encoder used the mobilenet_v2, 
    def __init__(self, feature_dim, num_classes, classifiers, shared_encoder, domain_encoder, domain_decoder):
        super(DSN_MobileNet, self).__init__()
        self.feature_dim = feature_dim
        self.n_classes = num_classes # the classification number of each task
        # self.classifiers = MultiTaskMimic

        # # source encoders for each task, why we need different domain encoders to learn domain knowledge?
        # self.domain_encoders = []
        # for i in range(num_classes):
        #     self.domain_encoders.append(domain_encoder)


        self.domain_encoder = domain_encoder
    
        # shared encoder to encoder all domains
        self.shared_encoder = shared_encoder
        
        # task specific classifiers and domain classifier 
        self.task_classifiers = classifiers(feature_dim, num_classes)

        # classify different domains
        self.domain_classifier.add_module('fc_se7', nn.Linear(in_features=feature_dim, out_features=tasks_num+1))

        # shared decoder ( decoder for reconstruct face image, use down-sample and  crop face(not 224*224 with background, only crop face region))
        self.shared_decoder = domain_decoder


    def forward(self, input_data, mode, rec_scheme, p=0.0):

        result = []

        
        # source private encoder
        private_feat = self.source_encoder_conv(input_data)
        private_feat = private_feat.view(-1, 64 * 7 * 7)
        private_code = self.source_encoder_fc(private_feat)


        # shared encoder
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 48 * 7 * 7)
        shared_code = self.shared_encoder_fc(shared_feat)
        result.append(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)

        if mode == 'source':
            class_label = self.shared_encoder_pred_class(shared_code)
            result.append(class_label)

        # shared decoder

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        rec_vec = self.shared_decoder_fc(union_code)
        rec_vec = rec_vec.view(-1, 3, 14, 14)

        rec_code = self.shared_decoder_conv(rec_vec)
        result.append(rec_code)

        return result





