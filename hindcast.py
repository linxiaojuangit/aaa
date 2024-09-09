#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from Network_Def import SmaAt_UNet
import torch
from torch import optim
from torch import nn
import numpy as np
import os
from config_args import get_args
from load_dataset import load_dataset
from loss_def import cal_rmse, cal_rmse_lead
# ---------------------------------------Parameter definition

args = get_args()

# ---------------------------------------Load dataset

valid_data = load_dataset( mode='valid')
valid_loader = DataLoader( dataset = valid_data, batch_size = args.BatchSize, num_workers = 4)

mask_index = valid_data._load_mask().to(args.device)
# ---------------------------------------Select the best model

loss = []
filename = './Record_RMSE_epoch.txt'
with open( filename, 'r' ) as fid:
    while True:
        aline = fid.readline()
        if not aline:
            break
        loss.append( float( aline.split( ',' )[4] ) )

loss = np.array( loss, dtype = np.float )
epoch_idx = np.argmin( loss )

print('np.argmin(loss): ', epoch_idx)
print('np.min(loss): ', loss[ epoch_idx ] )

# ---------------------------------------Optimization Configuration
model = SmaAt_UNet(n_channels = args.InChannel, n_classes = args.OutChannel)
model = nn.DataParallel(model)
model.to(args.device)

model.load_state_dict( torch.load( ( './data/Model_%s.pth' % str( epoch_idx ) ), map_location='cpu' ) )
print( 'Reading ' + ( './data/Model_%s.pth' % str( epoch_idx ) ) )

# ---------------------------------------Hindcast in all testing dataset 

model.eval()
with torch.no_grad():
    for ibatch, ( x, y ) in enumerate( valid_loader ):
        print(ibatch, len(valid_loader))         
        x = x.to(args.device)
        p = model(x).to(torch.device('cpu')).detach()

        p_col = p if ibatch == 0 else torch.cat((p_col, p), 0)
        y_col = y if ibatch == 0 else torch.cat((y_col, y), 0)

    rmse_lead_vld = cal_rmse_lead( y_col, p_col, mask_index.to(y_col.device)).numpy()
    rmse_vld = cal_rmse( y_col, p_col, mask_index.to(y_col.device)).item()

    np.save('./hindcast/rmse_vld.npy', rmse_vld)
    np.save('./hindcast/rmse_lead_vld.npy', rmse_lead_vld)
    np.save('./hindcast/fct_vld.npy', p_col.numpy())
    np.save('./hindcast/gth_vld.npy', y_col.numpy())


