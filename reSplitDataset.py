# -*- coding: utf-8 -*-

# this code used to re-split dataset into training and validing datasets of input training and testing set
# for example, the training and testing in input dataset is 80:20, we re-split it to 30:70. more validing, less training
# the format of the input datasets
import numpy as np
import argparse
import torch
import random


parser = argparse.ArgumentParser()
parser.add_argument('--training', type=str, required=True)
parser.add_argument('--testing' , type=str, required=True)
parser.add_argument('--training-percent', type=float, default=0.3)

args = parser.parse_args()

input_training = args.training
input_testing = args.testing

training_percent = args.training_percent

training_data = torch.load(input_training)
testing_data = torch.load(input_testing)

all_data  = torch.cat((training_data[0], testing_data[0]), 0)
all_label = torch.cat((training_data[1], testing_data[1]), 0)

print all_data.size()

num_data = all_label.size()[0]
indices = np.random.permutation(num_data)

training_index = indices[:int(num_data*training_percent)]
testing_index  = indices[int(num_data*training_percent):]

new_training = (all_data[training_index,:], all_label[training_index])
new_testing  = (all_data[testing_index,:], all_label[testing_index])

torch.save(new_training, 'mnist_training.pt')
torch.save(new_testing,  'mnist_testing.pt')