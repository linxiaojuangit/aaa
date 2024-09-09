import numpy as np
import datetime as dt 

fct = np.load('./hindcast/fct_vld.npy')#xiugai3
gth = np.load('./hindcast/gth_vld.npy')
dat = np.load('../data/data_samp/date_samp.npy')
msk = np.load('../data/data_clean/mask_clean.npy')
dat = dat[int(len(dat) * 0.8) :]
##DepregitcationWarning

nsmp, nlen, nx, ny = fct.shape
months = [1, 4, 7, 10]
idx_col = []
for ismp in range(nsmp):
    for mon in months:
        if (dat[ismp].astype(dt.datetime) + dt.timedelta(days = 7)) == dt.date(2021, mon, 1):
            idx_col.append(ismp)
np.save('./analysis/fct_slice.npy', fct[idx_col])
np.save('./analysis/gth_slice.npy', gth[idx_col])

raise()


# horizontal rmse
nsmp, nlen, nx, ny = fct.shape
rmse = np.ones((nlen, nx, ny), dtype = np.float32) * np.nan

for ilen in range(nlen):
    for ix in range(nx):
        for iy in range(ny):
             
            if msk[ix, iy] == 0:
                continue
             
            fct_sub = fct[:, ilen, ix, iy]
            gth_sub = gth[:, ilen, ix, iy]
            rmse[ilen, ix, iy] = np.sqrt(np.nanmean((fct_sub - gth_sub) ** 2))

np.save('./analysis/rmse_lead_hor.npy', rmse)

nsmp, nlen, nx, ny = fct.shape
rmse_dat = np.ones((nsmp, nlen), dtype = np.float32) * np.nan

for ismp in range(nsmp):
    for ilen in range(nlen):
        fct_sub = fct[ismp, ilen][msk == 1]
        gth_sub = gth[ismp, ilen][msk == 1]
        rmse_dat[ismp, ilen] = np.sqrt(np.nanmean((fct_sub - gth_sub) ** 2))

rmse_mon = np.ones((12, nlen), dtype = np.float32) * np.nan
for imon in range(12):
    for ilen in range(7):
        idx_col = []
        for ismp in range(nsmp):
            pmon = (dat[ismp].astype(dt.datetime) + dt.timedelta(days = 7 + ilen)).month
            if pmon - 1 == imon:
                idx_col.append(ismp)
        rmse_mon[imon, ilen] = np.mean(rmse_dat[idx_col, ilen])

np.save('./analysis/rmse_mon.npy', rmse_mon)
np.save('./analysis/rmse_dat.npy', rmse_dat)
