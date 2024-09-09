#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Network_Def import SmaAt_UNet
import torch
from torch import optim
from torch import nn
import torch.utils.data as DataSet
from torch.autograd import Variable
import numpy as np
import time
import os
from load_dataset import load_dataset
from config_args import get_args
from torch.utils.data import DataLoader
from loss_def import costfunction, cal_rmse, cal_rmse_lead

args = get_args() #xiugai

# ---------------------------------------Load dataset

train_data = load_dataset( mode = 'train' )
valid_data = load_dataset( mode = 'valid')

train_loader = DataLoader( dataset = train_data, batch_size = args.BatchSize, num_workers = 16, shuffle = True)
valid_loader = DataLoader( dataset = valid_data, batch_size = args.BatchSize, num_workers = 16)  #

mask_index = train_data._load_mask().to(args.device) 
# ---------------------------------------Optimization Configuration
model = SmaAt_UNet(n_channels = args.InChannel, n_classes = args.OutChannel)
model = nn.DataParallel(model)   #用来给GPU加速的
model.to(args.device)

opt = optim.Adam(model.parameters(), lr = args.LearningRate)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "max", factor = 0.1, patience = 4)

print('NetWork Initialization is done')

# ---------------------------------------Network Training
rmse_min = 1E10

for epoch in range( args.epochs ):
    model.train()
    time1_end = time.time()

    for ibatch, (x, y) in enumerate(train_loader):
        time1_sta = time.time()

        # [ Batch, Channel*InLen, Width, Height ] 
        x = x.to(args.device)
        y = y.to(args.device)
        # c = c.to(args.device)
        p = model(x)
        loss = costfunction(p, y, mask_index)

        opt.zero_grad()
        loss.backward()
        opt.step()
  
        time1_pre = time1_end
        time1_end = time.time()
        time_gpu = time1_end - time1_sta
        time_cpu = (time1_end - time1_pre) - time_gpu

        rmse_trn_batch = np.sqrt(loss.item()) 
        rmse_trn = [rmse_trn_batch] if ibatch == 0 else np.append(rmse_trn, [rmse_trn_batch])

        print('[%4d/%4d][%7d/%7d] RMSE, %15.10f, Time_gpu, %5.2f, Time_cpu, %5.2f' % \
              (epoch, args.epochs, ibatch, len(train_loader), rmse_trn_batch, time_gpu, time_cpu ) )

    rmse_trn = np.mean(rmse_trn)

    model.eval()
    with torch.no_grad():

        for ibatch, ( x, y ) in enumerate(valid_loader):
  
            x = x.to(args.device)
            p = model(x).to(torch.device('cpu')).detach()

            p_col = p if ibatch == 0 else torch.cat((p_col, p), 0)
            y_col = y if ibatch == 0 else torch.cat((y_col, y), 0)

        rmse_lead_vld = cal_rmse_lead( y_col, p_col, mask_index.to(y_col.device)).numpy()
        rmse_vld = cal_rmse( y_col, p_col, mask_index.to(y_col.device)).item()

        if rmse_vld <= rmse_min:
            rmse_min = rmse_vld
            torch.save(model.state_dict(), './data/Model_%s.pth' % str(epoch))

        fid = open('./Record_RMSE_epoch.txt', 'a')
        print('%4d, RMSE_trn, %10.5f, RMSE_vld, %10.5f, %8.6f, %8.6f, %8.6f, %8.6f, %8.6f, %8.6f, %8.6f' % \
              (epoch, rmse_trn, rmse_vld, *rmse_lead_vld.tolist()), file=fid)
        fid.close()
