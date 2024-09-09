#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from config_args import get_args

args = get_args()

# ---------------------------------------Functions

def costfunction(p, y, ocean_point_index):
    # [ Batch, Channel, Width, Height ] 
    msefunc = nn.MSELoss().to( p.device )

    y0 = y.reshape(y.shape[0], y.shape[1], -1)
    p0 = p.reshape(p.shape[0], p.shape[1], -1)
    y1 = torch.index_select(y0, 2, ocean_point_index)  #（input，纬度，指定的索引）
    p1 = torch.index_select(p0, 2, ocean_point_index)  #（输入的矩阵，指定读取第三维的数据，索引是读入的mask）
    loss = msefunc(p1, y1)
    return loss


def cal_rmse_lead(p, y, ocean_point_index):
    # [ samp, Channel, Width, Height ] 
                           # 输出长度
    rmse_lead = torch.empty( args.OutLen, dtype = torch.float ).to( p.device )

    for ilead in range( args.OutLen ):
        y_sub = y[:, ilead : ilead + 1 ]
        p_sub = p[:, ilead : ilead + 1 ]
        rmse_lead[ ilead ] = cal_rmse( y_sub, p_sub, ocean_point_index )
    return rmse_lead


def cal_rmse(p, y, ocean_point_index):
    # [ samp, Channel, Width, Height ] 
    return torch.sqrt( costfunction( p, y, ocean_point_index ) )


def main():
   mask = np.load('../data/data_clean/mask_clean.npy')
   mask_index = torch.tensor(np.where(mask.reshape(-1) == 1)[0])
   mask_index = mask_index.to(args.device) 

   p =  torch.ones([ 100, args.OutLen, 384, 320], dtype = torch.float )
   y = torch.zeros([ 100, args.OutLen, 384, 320], dtype = torch.float )

   p = p.to(args.device)
   y = y.to(args.device) 

   print( cal_rmse( p, y, mask_index ) )
   print( cal_rmse_lead( p, y, mask_index ) )
   print( costfunction( p, y, mask_index ) )

if __name__ == '__main__':
   main()


