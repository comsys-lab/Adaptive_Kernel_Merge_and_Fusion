#!/usr/bin/env python3
"""Export pretrained torchvision weights to the text files read by src/*.cu.

Usage: ./export_weights.py [alexnet] [resnet18] [vgg16]   (default: all three)

Writes data_alexnet/, data_resnet18/, data_vgg16/ in the current directory.
Each file holds one value per line. Conv weights are [out][in][kh][kw],
FC weights are [out][in] (PyTorch layout), and the input image is 224x224x3
in HWC order (RGB), normalized with the ImageNet mean/std.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

INPUT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "cat.jpg")


def save(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = tensor.detach().cpu().numpy().astype(np.float32).ravel()
    np.savetxt(path, arr, fmt="%.9g")


def input_image():
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(Image.open(INPUT_IMAGE).convert("RGB"))  # CHW
    return x.permute(1, 2, 0)                                # HWC


def export_alexnet():
    m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
    d = "data_alexnet"
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    fcs = [l for l in m.classifier if isinstance(l, torch.nn.Linear)]
    for i, l in enumerate(convs, 1):
        save(f"{d}/conv{i}.txt", l.weight)
        save(f"{d}/bias{i}.txt", l.bias)
    for i, l in enumerate(fcs, 6):
        save(f"{d}/fc{i}.txt", l.weight)
        save(f"{d}/bias{i}.txt", l.bias)
    save(f"{d}/input_cat1.txt", input_image())


def export_resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    d = "data_resnet18"
    # Layer numbering used in the CUDA code:
    #   1: stem conv, 2-17: the two convs of each BasicBlock in order,
    #   block3/4/5: 1x1 downsample convs of layer2/3/4.
    layers = [("1", m.conv1, m.bn1)]
    n = 2
    for stage in (m.layer1, m.layer2, m.layer3, m.layer4):
        for blk in stage:
            layers.append((str(n), blk.conv1, blk.bn1))
            layers.append((str(n + 1), blk.conv2, blk.bn2))
            n += 2
    for name, stage in (("block3", m.layer2), ("block4", m.layer3), ("block5", m.layer4)):
        layers.append((name, stage[0].downsample[0], stage[0].downsample[1]))

    for name, conv, bn in layers:
        sep = "_" if name.startswith("block") else ""
        save(f"{d}/conv_data/conv{sep}{name}.txt", conv.weight)
        save(f"{d}/gamma_data/gamma{sep}{name}.txt", bn.weight)
        save(f"{d}/beta_data/beta{sep}{name}.txt", bn.bias)
        save(f"{d}/mean_data/mean{sep}{name}.txt", bn.running_mean)
        save(f"{d}/var_data/var{sep}{name}.txt", bn.running_var)
    save(f"{d}/fc_data/fc1_weight.txt", m.fc.weight)
    save(f"{d}/fc_data/fc1_bias.txt", m.fc.bias)
    save(f"{d}/input_cat.txt", input_image())


def export_vgg16():
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval()
    d = "data_vgg16"
    convs = [l for l in m.features if isinstance(l, torch.nn.Conv2d)]
    fcs = [l for l in m.classifier if isinstance(l, torch.nn.Linear)]
    for i, l in enumerate(convs, 1):
        save(f"{d}/conv{i}.txt", l.weight)
        save(f"{d}/bias{i}.txt", l.bias)
    for i, l in enumerate(fcs, 14):
        save(f"{d}/fc{i}.txt", l.weight)
        save(f"{d}/bias{i}.txt", l.bias)
    save(f"{d}/input_cat.txt", input_image())


EXPORTERS = {"alexnet": export_alexnet, "resnet18": export_resnet18, "vgg16": export_vgg16}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(EXPORTERS)
    for t in targets:
        if t not in EXPORTERS:
            sys.exit(f"unknown model: {t} (choose from {', '.join(EXPORTERS)})")
    for t in targets:
        print(f"exporting {t} ...", flush=True)
        EXPORTERS[t]()
    print("done")
