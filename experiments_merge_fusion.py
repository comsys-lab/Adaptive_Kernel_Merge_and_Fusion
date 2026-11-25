#!/usr/bin/env python3
"""Adaptive Kernel Merge and Fusion (proposed).

The dispatcher looks at a window of MERGE_WINDOW (5) queued requests and groups
them by model:
  * kernel merging: requests of the same model are processed by one kernel
    launch that iterates over the requests;
  * kernel fusion: the merged AlexNet and ResNet18 kernels are fused, so one
    kernel computes a layer of both models (src/fused_kernels.cu).
VGG16 requests of the window are merged and run by a second process at the same
time as the fused AlexNet/ResNet18 kernels.

Usage (from the repository root):
    nvcc -Xcompiler -fPIC -shared -o merge_fusion.so src/merge_fusion.cu
    ./experiments_merge_fusion.py <num_requests> <arrival_rate>

<arrival_rate> is 95, 90, 85 or 80: requests arrive as a Poisson process whose
rate is that percentage of the service rate (1 / avg_inference_time).
The model of each request is read from random_model.txt.
Each processed request appends `<model> <queuing_delay> <inference_time>` to
results/merge_fusion.txt.
"""
import time, os
import sys
from math import factorial, exp
import multiprocessing
from multiprocessing import Process, Queue, Value
import numpy as np
import statistics as st
from numpy import random
import ctypes
from ctypes import *

# Dispatcher + AlexNet/ResNet18 process: takes a merge window of requests, sends the
# VGG16 ones to the VGG16 process, and runs the rest as merged and fused kernels.
def Alex_Res_inference(count, user_request, random_model, total, stopping, producer_status, finisher, vgg_queue, starttime_list, endtime_list):
    dll = ctypes.CDLL('./merge_fusion.so',mode=ctypes.RTLD_GLOBAL)
    Alex_Res_host2gpu = dll.Alex_Res_host2gpu
    Alex_Res_host2gpu.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                                POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]

    Alex_Res_inference = dll.Alex_Res_inference
    Alex_Res_inference.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                c_int,c_int]
                    
    Alex_Res_cudafree = dll.Alex_Res_cudafree
    Alex_Res_cudafree.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float)]

    
    A_L1_N = POINTER(c_float)()
    A_L2_N = POINTER(c_float)()
    A_L3_N = POINTER(c_float)()
    A_L4_N = POINTER(c_float)()
    A_L5_N = POINTER(c_float)()
    A_L6_N = POINTER(c_float)()
    A_L7_N = POINTER(c_float)()
    A_L8_N = POINTER(c_float)()

    A_L1_b = POINTER(c_float)()
    A_L2_b = POINTER(c_float)()
    A_L3_b = POINTER(c_float)()
    A_L4_b = POINTER(c_float)()
    A_L5_b = POINTER(c_float)()
    A_L6_b = POINTER(c_float)()
    A_L7_b = POINTER(c_float)()
    A_L8_b = POINTER(c_float)()

    A_L1_w = POINTER(c_float)()
    A_L2_w = POINTER(c_float)()
    A_L3_w = POINTER(c_float)()
    A_L4_w = POINTER(c_float)()
    A_L5_w = POINTER(c_float)()
    A_L6_w = POINTER(c_float)()
    A_L7_w = POINTER(c_float)()
    A_L8_w = POINTER(c_float)()

    A_L1_pool = POINTER(c_float)()
    A_L2_pool = POINTER(c_float)()
    A_L5_pool = POINTER(c_float)()
    A_L1_norm = POINTER(c_float)()
    A_L2_norm = POINTER(c_float)()
    A_Result_N = POINTER(c_float)()

    R_L1_N = POINTER(c_float)()
    R_L2_N = POINTER(c_float)()
    R_L3_N = POINTER(c_float)()
    R_L4_N = POINTER(c_float)()
    R_L5_N = POINTER(c_float)()
    R_L6_N = POINTER(c_float)()
    R_L7_N = POINTER(c_float)()
    R_L8_N = POINTER(c_float)()
    R_L9_N = POINTER(c_float)()
    R_L10_N = POINTER(c_float)()
    R_L11_N = POINTER(c_float)()
    R_L12_N = POINTER(c_float)()
    R_L13_N = POINTER(c_float)()
    R_L14_N = POINTER(c_float)()
    R_L15_N = POINTER(c_float)()
    R_L16_N = POINTER(c_float)()
    R_L17_N = POINTER(c_float)()
    R_L18_N = POINTER(c_float)()

    R_L1_w = POINTER(c_float)()
    R_L2_w = POINTER(c_float)()
    R_L3_w = POINTER(c_float)()
    R_L4_w = POINTER(c_float)()
    R_L5_w = POINTER(c_float)()
    R_L6_w = POINTER(c_float)()
    R_L7_w = POINTER(c_float)()
    R_L8_w = POINTER(c_float)()
    R_L9_w = POINTER(c_float)()
    R_L10_w = POINTER(c_float)()
    R_L11_w = POINTER(c_float)()
    R_L12_w = POINTER(c_float)()
    R_L13_w = POINTER(c_float)()
    R_L14_w = POINTER(c_float)()
    R_L15_w = POINTER(c_float)()
    R_L16_w = POINTER(c_float)()
    R_L17_w = POINTER(c_float)()
    R_B3_w = POINTER(c_float)()
    R_B4_w = POINTER(c_float)()
    R_B5_w = POINTER(c_float)()

    R_L1_G = POINTER(c_float)()
    R_L2_G = POINTER(c_float)()
    R_L3_G = POINTER(c_float)()
    R_L4_G = POINTER(c_float)()
    R_L5_G = POINTER(c_float)()
    R_L6_G = POINTER(c_float)()
    R_L7_G = POINTER(c_float)()
    R_L8_G = POINTER(c_float)()
    R_L9_G = POINTER(c_float)()
    R_L10_G = POINTER(c_float)()
    R_L11_G = POINTER(c_float)()
    R_L12_G = POINTER(c_float)()
    R_L13_G = POINTER(c_float)()
    R_L14_G = POINTER(c_float)()
    R_L15_G = POINTER(c_float)()
    R_L16_G = POINTER(c_float)()
    R_L17_G = POINTER(c_float)()
    R_B3_G = POINTER(c_float)()
    R_B4_G = POINTER(c_float)()
    R_B5_G = POINTER(c_float)()

    R_L1_B = POINTER(c_float)()
    R_L2_B = POINTER(c_float)()
    R_L3_B = POINTER(c_float)()
    R_L4_B = POINTER(c_float)()
    R_L5_B = POINTER(c_float)()
    R_L6_B = POINTER(c_float)()
    R_L7_B = POINTER(c_float)()
    R_L8_B = POINTER(c_float)()
    R_L9_B = POINTER(c_float)()
    R_L10_B = POINTER(c_float)()
    R_L11_B = POINTER(c_float)()
    R_L12_B = POINTER(c_float)()
    R_L13_B = POINTER(c_float)()
    R_L14_B = POINTER(c_float)()
    R_L15_B = POINTER(c_float)()
    R_L16_B = POINTER(c_float)()
    R_L17_B = POINTER(c_float)()
    R_B3_B = POINTER(c_float)()
    R_B4_B = POINTER(c_float)()
    R_B5_B = POINTER(c_float)()

    R_L1_M = POINTER(c_float)()
    R_L2_M = POINTER(c_float)()
    R_L3_M = POINTER(c_float)()
    R_L4_M = POINTER(c_float)()
    R_L5_M = POINTER(c_float)()
    R_L6_M = POINTER(c_float)()
    R_L7_M = POINTER(c_float)()
    R_L8_M = POINTER(c_float)()
    R_L9_M = POINTER(c_float)()
    R_L10_M = POINTER(c_float)()
    R_L11_M = POINTER(c_float)()
    R_L12_M = POINTER(c_float)()
    R_L13_M = POINTER(c_float)()
    R_L14_M = POINTER(c_float)()
    R_L15_M = POINTER(c_float)()
    R_L16_M = POINTER(c_float)()
    R_L17_M = POINTER(c_float)()
    R_B3_M = POINTER(c_float)()
    R_B4_M = POINTER(c_float)()
    R_B5_M = POINTER(c_float)()

    R_L1_V = POINTER(c_float)()
    R_L2_V = POINTER(c_float)()
    R_L3_V = POINTER(c_float)()
    R_L4_V = POINTER(c_float)()
    R_L5_V = POINTER(c_float)()
    R_L6_V = POINTER(c_float)()
    R_L7_V = POINTER(c_float)()
    R_L8_V = POINTER(c_float)()
    R_L9_V = POINTER(c_float)()
    R_L10_V = POINTER(c_float)()
    R_L11_V = POINTER(c_float)()
    R_L12_V = POINTER(c_float)()
    R_L13_V = POINTER(c_float)()
    R_L14_V = POINTER(c_float)()
    R_L15_V = POINTER(c_float)()
    R_L16_V = POINTER(c_float)()
    R_L17_V = POINTER(c_float)()
    R_B3_V = POINTER(c_float)()
    R_B4_V = POINTER(c_float)()
    R_B5_V = POINTER(c_float)()

    R_FC_b = POINTER(c_float)()
    R_FC_w = POINTER(c_float)()

    R_L3_basic = POINTER(c_float)()
    R_L5_basic = POINTER(c_float)()
    R_L7_basic = POINTER(c_float)()
    R_L9_basic = POINTER(c_float)()
    R_L11_basic = POINTER(c_float)()
    R_L13_basic = POINTER(c_float)()
    R_L15_basic = POINTER(c_float)()
    R_L17_basic = POINTER(c_float)()
    R_B3_basic = POINTER(c_float)()
    R_B4_basic = POINTER(c_float)()
    R_B5_basic = POINTER(c_float)()
    R_L1_bn = POINTER(c_float)()
    R_L2_bn = POINTER(c_float)()
    R_L3_bn = POINTER(c_float)()
    R_L4_bn = POINTER(c_float)()
    R_L5_bn = POINTER(c_float)()
    R_L6_bn = POINTER(c_float)()
    R_L7_bn = POINTER(c_float)()
    R_L8_bn = POINTER(c_float)()
    R_L9_bn = POINTER(c_float)()
    R_L10_bn = POINTER(c_float)()
    R_L11_bn = POINTER(c_float)()
    R_L12_bn = POINTER(c_float)()
    R_L13_bn = POINTER(c_float)()
    R_L14_bn = POINTER(c_float)()
    R_L15_bn = POINTER(c_float)()
    R_L16_bn = POINTER(c_float)()
    R_L17_bn = POINTER(c_float)()
    R_B3_bn = POINTER(c_float)()
    R_B4_bn = POINTER(c_float)()
    R_B5_bn = POINTER(c_float)()
    R_L1_pool = POINTER(c_float)()
    R_FC_N = POINTER(c_float)()
    R_Result_N = POINTER(c_float)()

    Alex_Res_host2gpu(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N,
                    R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

    # Warm-up runs (not measured)
    for i in range(10):
        Alex_Res_inference(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                            A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                            A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                            A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                            A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                            A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                            A_L1_pool,A_L2_pool,A_L5_pool,
                            A_L1_norm,A_L2_norm,A_Result_N,
                            R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                            R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                            R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                            R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                            R_L17_N,R_L18_N,
                            R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                            R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                            R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                            R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                            R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                            R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                            R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                            R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                            R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                            R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                            R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                            R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                            R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                            R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                            R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                            R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                            R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                            R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                            R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                            R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                            R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                            R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                            R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                            R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                            R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                            R_FC_b,R_FC_w,
                            R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                            R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                            R_B3_basic,R_B4_basic,R_B5_basic,
                            R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                            R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                            R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                            R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                            R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                            R_L1_pool,R_FC_N,R_Result_N,
                            1,1)

    # Signal that this process is ready; all processes start together.
    stopping.value = stopping.value + 1

    print("Alex&Res Inference Ready!")
    while(stopping.value < 2):
        continue

    total_alex_num = 0
    total_res_num = 0
    dispatched = 0      # requests taken from the queue so far

    starttime = time.time()
    starttime_list.append(starttime)
    while dispatched < total:
        # Wait until a full merge window of requests is queued. Once the request
        # generator has finished, process whatever is left in the queue.
        if user_request.qsize() < MERGE_WINDOW:
            if producer_status.value == 0:
                continue
            if user_request.empty():
                break

        # Take one merge window and group the requests by model.
        n = min(MERGE_WINDOW, user_request.qsize(), total - dispatched)
        dispatch_time = time.time()
        alex_num = 0
        res_num = 0
        alex_res_delays = []    # queuing delays of the AlexNet/ResNet18 requests
        vgg_arrivals = []       # arrival times of the VGG16 requests
        for i in range(n):
            model = random_model.pop(0)
            arrival_time = user_request.get()
            if model == 'alexnet':
                alex_num += 1
                alex_res_delays.append(max(0.0, dispatch_time - arrival_time))
            elif model == 'resnet18':
                res_num += 1
                alex_res_delays.append(max(0.0, dispatch_time - arrival_time))
            else:
                vgg_arrivals.append(arrival_time)
        dispatched += n

        # VGG16 requests of this window are merged and run by the VGG16 process.
        if vgg_arrivals:
            vgg_queue.put(vgg_arrivals)

        # AlexNet and ResNet18 requests are merged per model, and the two merged
        # models are fused into one set of kernels.
        model_num = alex_num + res_num
        if model_num > 0:
            inference_start_time = time.time()
            Alex_Res_inference(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                                A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                                A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                                A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                                A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                                A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                                A_L1_pool,A_L2_pool,A_L5_pool,
                                A_L1_norm,A_L2_norm,A_Result_N,
                                R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                                R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                                R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                                R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                                R_L17_N,R_L18_N,
                                R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                                R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                                R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                                R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                                R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                                R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                                R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                                R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                                R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                                R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                                R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                                R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                                R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                                R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                                R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                                R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                                R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                                R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                                R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                                R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                                R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                                R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                                R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                                R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                                R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                                R_FC_b,R_FC_w,
                                R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                                R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                                R_B3_basic,R_B4_basic,R_B5_basic,
                                R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                                R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                                R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                                R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                                R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                                R_L1_pool,R_FC_N,R_Result_N,
                                alex_num,res_num)
            inference_end_time = time.time()
            inference_time = (inference_end_time-inference_start_time)/model_num
            with open("results/merge_fusion.txt", "a") as file:
                for q_d in alex_res_delays:
                    file.write("Fused model(Alex:{} ,Res:{})" f"\t{(q_d):.6f} \t {(inference_time):.6f}\n".format(alex_num,res_num))
            total_alex_num += alex_num
            total_res_num += res_num
            for i in range(model_num):
                count.put(0)

    vgg_queue.put(None)     # tell the VGG16 process that no more requests come
    finisher.value = 1      # stop the request generator
    endtime = time.time()
    endtime_list.append(endtime)
    print(f"Alenet & Resnet18 inference time : {(endtime-starttime):.6f} seconds")
    print("Total Alexnet: {}".format(total_alex_num))
    print("Total Resnet18: {}".format(total_res_num))

    Alex_Res_cudafree(A_L1_N,A_L2_N,A_L3_N,A_L4_N,
                    A_L5_N,A_L6_N,A_L7_N,A_L8_N,
                    A_L1_b,A_L2_b,A_L3_b,A_L4_b,
                    A_L5_b,A_L6_b,A_L7_b,A_L8_b,
                    A_L1_w,A_L2_w,A_L3_w,A_L4_w,
                    A_L5_w,A_L6_w,A_L7_w,A_L8_w,
                    A_L1_pool,A_L2_pool,A_L5_pool,
                    A_L1_norm,A_L2_norm,A_Result_N,
                    R_L1_N,R_L2_N,R_L3_N,R_L4_N,
                    R_L5_N,R_L6_N,R_L7_N,R_L8_N,
                    R_L9_N,R_L10_N,R_L11_N,R_L12_N,
                    R_L13_N,R_L14_N,R_L15_N,R_L16_N,
                    R_L17_N,R_L18_N,
                    R_L1_w,R_L2_w,R_L3_w,R_L4_w,
                    R_L5_w,R_L6_w,R_L7_w,R_L8_w,
                    R_L9_w,R_L10_w,R_L11_w,R_L12_w,
                    R_L13_w,R_L14_w,R_L15_w,R_L16_w,
                    R_L17_w,R_B3_w,R_B4_w,R_B5_w,
                    R_L1_G,R_L2_G,R_L3_G,R_L4_G,
                    R_L5_G,R_L6_G,R_L7_G,R_L8_G,
                    R_L9_G,R_L10_G,R_L11_G,R_L12_G,
                    R_L13_G,R_L14_G,R_L15_G,R_L16_G,
                    R_L17_G,R_B3_G,R_B4_G,R_B5_G,
                    R_L1_B,R_L2_B,R_L3_B,R_L4_B,
                    R_L5_B,R_L6_B,R_L7_B,R_L8_B,
                    R_L9_B,R_L10_B,R_L11_B,R_L12_B,
                    R_L13_B,R_L14_B,R_L15_B,R_L16_B,
                    R_L17_B,R_B3_B,R_B4_B,R_B5_B,
                    R_L1_M,R_L2_M,R_L3_M,R_L4_M,
                    R_L5_M,R_L6_M,R_L7_M,R_L8_M,
                    R_L9_M,R_L10_M,R_L11_M,R_L12_M,
                    R_L13_M,R_L14_M,R_L15_M,R_L16_M,
                    R_L17_M,R_B3_M,R_B4_M,R_B5_M,
                    R_L1_V,R_L2_V,R_L3_V,R_L4_V,
                    R_L5_V,R_L6_V,R_L7_V,R_L8_V,
                    R_L9_V,R_L10_V,R_L11_V,R_L12_V,
                    R_L13_V,R_L14_V,R_L15_V,R_L16_V,
                    R_L17_V,R_B3_V,R_B4_V,R_B5_V,
                    R_FC_b,R_FC_w,
                    R_L3_basic,R_L5_basic,R_L7_basic,R_L9_basic,
                    R_L11_basic,R_L13_basic,R_L15_basic,R_L17_basic,
                    R_B3_basic,R_B4_basic,R_B5_basic,
                    R_L1_bn,R_L2_bn,R_L3_bn,R_L4_bn,
                    R_L5_bn,R_L6_bn,R_L7_bn,R_L8_bn,
                    R_L9_bn,R_L10_bn,R_L11_bn,R_L12_bn,
                    R_L13_bn,R_L14_bn,R_L15_bn,R_L16_bn,
                    R_L17_bn,R_B3_bn,R_B4_bn,R_B5_bn,
                    R_L1_pool,R_FC_N,R_Result_N)

# VGG16 process: runs each batch of VGG16 requests from the dispatcher as merged kernels.
def Vgg_merge_inference(count, user_request, random_model, total, stopping, producer_status, finisher, vgg_queue, starttime_list, endtime_list):
    dll = ctypes.CDLL('./merge_fusion.so',mode=ctypes.RTLD_GLOBAL)
    Vgg_host2gpu = dll.Vgg_host2gpu
    Vgg_host2gpu.argtypes = [POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),POINTER(POINTER(c_float)),
                            POINTER(POINTER(c_float)),POINTER(POINTER(c_float))]
    
    Vgg_merge_inference = dll.Vgg_merge_inference
    Vgg_merge_inference.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                                POINTER(c_float),POINTER(c_float),c_int]

    Vgg_cudafree = dll.Vgg_cudafree
    Vgg_cudafree.argtypes = [POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float),POINTER(c_float),POINTER(c_float),
                            POINTER(c_float),POINTER(c_float)]

    V_L1_N = POINTER(c_float)()
    V_L2_N = POINTER(c_float)()
    V_L3_N = POINTER(c_float)()
    V_L4_N = POINTER(c_float)()
    V_L5_N = POINTER(c_float)()
    V_L6_N = POINTER(c_float)()
    V_L7_N = POINTER(c_float)()
    V_L8_N = POINTER(c_float)()
    V_L9_N = POINTER(c_float)()
    V_L10_N = POINTER(c_float)()
    V_L11_N = POINTER(c_float)()
    V_L12_N = POINTER(c_float)()
    V_L13_N = POINTER(c_float)()
    V_L14_N = POINTER(c_float)()
    V_L15_N = POINTER(c_float)()
    V_L16_N = POINTER(c_float)()

    V_L1_b = POINTER(c_float)()
    V_L2_b = POINTER(c_float)()
    V_L3_b = POINTER(c_float)()
    V_L4_b = POINTER(c_float)()
    V_L5_b = POINTER(c_float)()
    V_L6_b = POINTER(c_float)()
    V_L7_b = POINTER(c_float)()
    V_L8_b = POINTER(c_float)()
    V_L9_b = POINTER(c_float)()
    V_L10_b = POINTER(c_float)()
    V_L11_b = POINTER(c_float)()
    V_L12_b = POINTER(c_float)()
    V_L13_b = POINTER(c_float)()
    V_L14_b = POINTER(c_float)()
    V_L15_b = POINTER(c_float)()
    V_L16_b = POINTER(c_float)()

    V_L1_w = POINTER(c_float)()
    V_L2_w = POINTER(c_float)()
    V_L3_w = POINTER(c_float)()
    V_L4_w = POINTER(c_float)()
    V_L5_w = POINTER(c_float)()
    V_L6_w = POINTER(c_float)()
    V_L7_w = POINTER(c_float)()
    V_L8_w = POINTER(c_float)()
    V_L9_w = POINTER(c_float)()
    V_L10_w = POINTER(c_float)()
    V_L11_w = POINTER(c_float)()
    V_L12_w = POINTER(c_float)()
    V_L13_w = POINTER(c_float)()
    V_L14_w = POINTER(c_float)()
    V_L15_w = POINTER(c_float)()
    V_L16_w = POINTER(c_float)()

    V_L2_pool = POINTER(c_float)()
    V_L4_pool = POINTER(c_float)()
    V_L7_pool = POINTER(c_float)()
    V_L10_pool = POINTER(c_float)()
    V_L13_pool = POINTER(c_float)()
    V_Result_N = POINTER(c_float)()

    Vgg_host2gpu(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                    V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                    V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                    V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                    V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                    V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                    V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                    V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                    V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                    V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                    V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                    V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                    V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                    V_L13_pool,V_Result_N)

    # Warm-up runs (not measured)
    for i in range(10):
        Vgg_merge_inference(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                            V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                            V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                            V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                            V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                            V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                            V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                            V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                            V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                            V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                            V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                            V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                            V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                            V_L13_pool,V_Result_N,1)

    # Signal that this process is ready; all processes start together.
    stopping.value = stopping.value + 1

    print("Vgg Inference Ready!")
    while(stopping.value < 2):
        continue

    total_vgg_num = 0

    starttime = time.time()
    starttime_list.append(starttime)
    while True:
        # Each item is the list of arrival times of the VGG16 requests in one
        # merge window. None means the dispatcher has finished.
        vgg_arrivals = vgg_queue.get()
        if vgg_arrivals is None:
            break
        vgg_num = len(vgg_arrivals)
        inference_start_time = time.time()
        Vgg_merge_inference(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                            V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                            V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                            V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                            V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                            V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                            V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                            V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                            V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                            V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                            V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                            V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                            V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                            V_L13_pool,V_Result_N,vgg_num)
        inference_end_time = time.time()
        inference_time = (inference_end_time-inference_start_time)/vgg_num
        with open("results/merge_fusion.txt", "a") as file:
            for arrival_time in vgg_arrivals:
                queueing_delay = max(0.0, inference_start_time - arrival_time)
                file.write("Fused model(Vgg:{})" f"\t{(queueing_delay):.6f} \t {(inference_time):.6f}\n".format(vgg_num))
        total_vgg_num += vgg_num
        for i in range(vgg_num):
            count.put(0)
    endtime = time.time()
    endtime_list.append(endtime)
    print(f"Vgg16 inference time : {(endtime-starttime):.6f} seconds")
    print("Total Vgg16: {}".format(total_vgg_num))

    Vgg_cudafree(V_L1_N,V_L2_N,V_L3_N,V_L4_N,
                V_L5_N,V_L6_N,V_L7_N,V_L8_N,
                V_L9_N,V_L10_N,V_L11_N,V_L12_N,
                V_L13_N,V_L14_N,V_L15_N,V_L16_N,
                V_L1_b,V_L2_b,V_L3_b,V_L4_b,
                V_L5_b,V_L6_b,V_L7_b,V_L8_b,
                V_L9_b,V_L10_b,V_L11_b,V_L12_b,
                V_L13_b,V_L14_b,V_L15_b,V_L16_b,
                V_L1_w,V_L2_w,V_L3_w,V_L4_w,
                V_L5_w,V_L6_w,V_L7_w,V_L8_w,
                V_L9_w,V_L10_w,V_L11_w,V_L12_w,
                V_L13_w,V_L14_w,V_L15_w,V_L16_w,
                V_L2_pool,V_L4_pool,V_L7_pool,V_L10_pool,
                V_L13_pool,V_Result_N)
    
# Request generator: puts the arrival time of each request into the queue,
# with Poisson-distributed intervals. Requests are dropped when the queue is full.
def request_arrival(user_request, queue_size, arrival_time, stopping, producer_status, finisher):
    pid = os.getpid()
    print('request producer PID: ',pid)

    # count1 = 0
    # count2 = 0
    # count3 = 0

    not_processed = 0
    poisson_list = []

    time.sleep(2)  

    while(stopping.value < 2):
        continue
    
    print("request creator start!")
    not_processed = 0

    _lambda = arrival_time
    mod_lambda = _lambda * 1000000                 

    poisson_list = random.poisson(mod_lambda, 2000)       
    poisson_list = poisson_list / 1000000 

    for timeslot in poisson_list:
        time.sleep (timeslot)
        requested_time = time.time()
        if(user_request.qsize() <= queue_size and finisher.value != 1):                            
            user_request.put(requested_time)
        elif(finisher.value == 1):
            break                                           # all requests processed: stop generating
        else:
            not_processed += 1
            continue 
    producer_status.value = 1
    # print("Request models: Alex: {}, Res: {}, Vgg: {}\n".format(count1,count2,count3))
    print(pid, ": request producer finish")
    print('{0} requests not added to request queue'.format(not_processed))

USAGE = "usage: {} <num_requests> <arrival_rate: 95|90|85|80>".format(sys.argv[0])

def get_total_requests():
    # Number of requests to process
    if len(sys.argv) != 3:
        sys.exit(USAGE)
    return int(sys.argv[1])

def get_arrival_time(avg_inference_time):
    # Mean arrival interval so that the arrival rate is N% of the service rate
    rate = int(sys.argv[2])
    if rate not in (95, 90, 85, 80):
        sys.exit(USAGE)
    return avg_inference_time*(100/rate)


stopping = Value('i', 0)  #for simultaneous starting of inferences between processes
producer_status = Value('i', 0)     #if request producer is done, turn into 1
finisher = Value('i', 0)     #if request producer is done, turn into 1
MERGE_WINDOW = 5   # number of queued requests examined per merge decision

if __name__ == '__main__':

    user_request = Queue()
    vgg_queue = Queue()     # merged VGG16 batches: dispatcher -> VGG16 process
    count = Queue()

    manager = multiprocessing.Manager()
    starttime_list = manager.list()
    endtime_list = manager.list()

    random_model = manager.list()
    with open("random_model.txt") as f:
        for line in f:
            random_model.append(line.strip()) 

    # Average inference time (s) of AlexNet, ResNet18 and VGG16 measured on Jetson
    # while running concurrently with request intervals. It sets the service rate,
    # and the same value is used by every policy so that all see the same arrivals.
    avg_inference_time = 0.076218889

    total = get_total_requests()
    arrival_time = get_arrival_time(avg_inference_time)
     
    rho = avg_inference_time/arrival_time
    queue_size = round((rho*rho)/(1-rho))
    # The merge window takes 5 requests at a time, so the queue must be able to hold 5.
    # Without this, the 80% arrival rate gives queue_size = 3 and the run never starts.
    queue_size = max(queue_size, 5)

    children = []

    start_time = time.time()
   
    children.append(Process(target = Alex_Res_inference, args=(count, user_request, random_model, total, stopping, producer_status, finisher, vgg_queue, starttime_list, endtime_list)))
    children.append(Process(target = Vgg_merge_inference, args=(count, user_request, random_model, total, stopping, producer_status, finisher, vgg_queue, starttime_list, endtime_list)))
    children.append(Process(target = request_arrival, args = (user_request, queue_size, arrival_time, stopping, producer_status, finisher)))

    for child in children:
        child.start()

    for child in children:
        child.join()

    starttime = min(starttime_list)
    endtime = max(endtime_list)
    total_inference_time = endtime-starttime
    end_time = time.time()
    total_time = end_time-start_time
    
    print(queue_size)
    print(f"total inferene time : {(total_inference_time):.6f}")
    print(f"total time : {(end_time-start_time):.6f}")
    print('total processed requests : {0}'.format(count.qsize()))

