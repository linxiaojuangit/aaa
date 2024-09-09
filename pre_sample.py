import h5py
import numpy as np
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime


fsst=np.load('./data_clean_con/fsst_series.npy')
rsst=np.load('./data_clean_con/rsst_series.npy')
date0=np.load('./data_clean_con/ftime_series.npy')
csst=np.load('./data_clean_con/con_sst_series.npy')


#Normalization
def normal_data(rsst):

    rmean=np.nanmean(np.nanmean(np.nanmean(rsst,axis=0, dtype=rsst.dtype),axis=0, dtype=rsst.dtype),axis=0, dtype=rsst.dtype)

    rstd = np.nanstd(rsst, dtype=rsst.dtype)

    return rmean,rstd

def nomalization(fsst,rsst,csst):

    rmean,rstd = normal_data(rsst)
    print(rmean,rstd)
    fsst_norm = (fsst-rmean)/rstd
    rsst_norm = (rsst-rmean)/rstd
    csst_norm = (csst-rmean)/rstd

    return fsst_norm,rsst_norm,csst_norm

fsst,rsst,csst = nomalization(fsst,rsst,csst)


nlen = 8
InLen = 7  #输入长度
OutLen = 1  #输出长度
ntim, ndim0, ndim1 = fsst.shape

# clim = create_clm_cycle(fsst, date0)    #月平均数�?
x_col = []
y_col = []
cy_col = []
y_clm = []
date_col = []
date0 = date0.astype('str')
date0 = date0.astype(dt.datetime)
for itim in range(ntim):
    if itim + nlen - 1 >= ntim:  #读取数字超过长度则停�?
        break

    pd0 = datetime.strptime(date0[itim],'%Y-%m-%d')+ relativedelta(months=nlen-1)#nlen-1
    pd0 = pd0.strftime('%Y%m')
    # print(pd0)
    pd1 = datetime.strptime( date0[itim + nlen - 1],'%Y-%m-%d')
    pd1 = pd1.strftime('%Y%m')
    # print(pd1)
    if pd0 == pd1:
        x_col.append(fsst[itim : itim + InLen])
        y_col.append(rsst[itim + InLen : itim + InLen + OutLen])
        cy_col.append(csst[itim + InLen : itim + InLen + OutLen])
        # y_clm.append(clim[itim + InLen : itim + InLen + OutLen])  #同时包含海冰>和时间的
        date_col.append(date0[itim])

date_col = np.array(date_col, dtype = np.datetime64)
np.save('./data_samp/date_samp.npy', date_col)

nsmp = len(x_col)
x_trn = x_col[: int(nsmp * 0.8)]  #20%作为测试数据
y_trn = y_col[: int(nsmp * 0.8)]
# c_trn = cy_col[: int(nsmp * 0.8)]
x_vld = x_col[int(nsmp * 0.8) :]
y_vld = y_col[int(nsmp * 0.8) :]
# c_vld = cy_col[int(nsmp * 0.8) :]

# y_trn_clm = y_clm[: int(nsmp * 0.8)]  #每天的气候态数据，这里是每月
# y_vld_clm = y_clm[int(nsmp * 0.8) :]

with h5py.File('./data_samp/icec_samples.h5', 'w') as hf:

    for isamp in range(len(x_trn)):   #把数据和标签一一对应起来
        cnt_samp_str = 'trn_%04d' % isamp
        key_x = '%s/X' % cnt_samp_str
        key_y = '%s/Y' % cnt_samp_str
        # key_c = '%s/C' % cnt_samp_str
        hf.create_dataset(key_x, data = x_trn[isamp])
        hf.create_dataset(key_y, data = y_trn[isamp])
        # hf.create_dataset(key_c, data = c_trn[isamp])

    for isamp in range(len(x_vld)):
        cnt_samp_str = 'vld_%04d' % isamp
        key_x = '%s/X' % cnt_samp_str
        key_y = '%s/Y' % cnt_samp_str
        # key_c = '%s/C' % cnt_samp_str
        hf.create_dataset(key_x, data = x_vld[isamp])
        hf.create_dataset(key_y, data = y_vld[isamp])
        # hf.create_dataset(key_c, data = c_vld[isamp])

    # for isamp in range(len(y_trn_clm)):
    #     cnt_samp_str = 'trn_clm_%04d' % isamp
    #     key_y = '%s/Y' % cnt_samp_str
    #     hf.create_dataset(key_y, data = y_trn_clm[isamp])

    # for isamp in range(len(y_vld_clm)):
    #     cnt_samp_str = 'vld_clm_%04d' % isamp
    #     key_y = '%s/Y' % cnt_samp_str
    #     hf.create_dataset(key_y, data = y_vld_clm[isamp])




                        
