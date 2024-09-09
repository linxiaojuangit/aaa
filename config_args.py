#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch
# ---------------------------------------Parameter definition

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--BatchSize', type=int, help='batch size', default = 16)
    parser.add_argument('--epochs', type=int, help='number of epochs', default = 1000)
    parser.add_argument('--LearningRate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--InLen', type=int, help='length of time slice of input', default = 7)
    parser.add_argument('--OutLen', type=int, help='length of time slice of output', default = 1)#7
    parser.add_argument('--Img_Width', type=int, help='width of image', default = 962)
    parser.add_argument('--Img_Height', type=int, help='height of image', default = 700)
    parser.add_argument('--device', type=float, help='use cuda or not', default = torch.device('cuda'))
    parser.add_argument('--InChannel', type=int, help='Num of Channel of input', default = 14 )
    parser.add_argument('--OutChannel', type=int, help='Num of Channel of output', default = 7 )
    args, unknown = parser.parse_known_args()

    return args
