#!/usr/bin/env python3
"""Generate random_model.txt: the model of each of 2000 requests, chosen uniformly
at random from AlexNet, ResNet18 and VGG16.

Running this script overwrites the random_model.txt in the repository with a new
random order.
"""
from numpy import random

NUM_REQUESTS = 2000

with open("random_model.txt", "w") as f:
    for i in range(NUM_REQUESTS):
        f.write(random.choice(['alexnet', 'resnet18', 'vgg16']) + '\n')
