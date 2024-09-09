#!usr/bin/python
# coding = utf-8

### 读取观测预报数据，时间序列一致，网格一致，单位一致	fde567b	linxiaojuangit <959393813@qq.com>	2024年8月31日 19:45

from pickletools import read_float8
# from tkinter.messagebox import RETRY
# import re
import numpy as np
# import numba as nb
# import xarray as xr
import os
import pandas as pd
import netCDF4 as nc
import datetime as dt
from glob import glob
from scipy import interpolate
from scipy.interpolate import griddata
from dateutil.relativedelta import relativedelta

#读入纬向速度UVEL，读入混合层深度XMXL--h
def read_forecast_data(pmns):
    # global
    fore_sst = []
    flat = []
    flon = []
    # pmns = int(pmns)
    # pmns = dt.date(pmns)
    # pmns=pmns.astype(dt.datetime)
    # pmns_dt = pmns+relativedelta(days=1)
    # print(pmns_dt)
    fname = '/Users/liuchunda/Desktop/project/train_model/macom_data/%s/MaCOM_daily_%s_t_7day.nc'  % (pmns.strftime('%Y%m%d'),pmns.strftime('%Y%m%d'))
    # fname = '/work/person/zuzq/EnOI_IAU/OCN_ASSIM/%s/EXP*.pop.h.%s.nc' % (pmns.strftime('%Y%m'),pmns_dt.strftime('%Y-%m'))
    #路径中有些没有规律的部分，可以用星号代替，然后用glob补齐路径
    fname_e =glob(fname)
    # print(fname_e)
    if len(fname_e) > 0:
        fpath = fname_e[0]
        # print(fpath)

        if os.path.exists(fpath):
            fid1 = nc.Dataset(fpath,mode="r" ,format="NETCDF4")
            # fore_sst = fid1.variables['SST'][0,:, :]
            fore_sst = fid1.variables['t'][0,0,:, :] ########!!!!!!这里改过一次！！！！！！！！
            flat = fid1.variables['lat'][:] #二维
            flon = fid1.variables['lon'][:]
 # if flat[0] > flat[-1]:  #维度是不是颠倒的，如果颠倒要正过来
            #     flat = np.flipud(flat)
            #     fore_sst = np.flipud(fore_sst)
            # fore_sst= np.flip(fore_sst,axis=0)
            # fore_sst = fore_sst.copy()
            # fore_sst = np.ascontiguousarray(fore_sst)
            # flat= np.flip(flat,axis=0)
            # flat = flat.copy()
            # flat = np.ascontiguousarray(flat)
            # print(flat)
            fore_sst[fore_sst>50] = np.nan
            fore_sst = np.array(fore_sst,dtype = np.float32)
            flat = np.array(flat,dtype = np.float32)
            flon = np.array(flon,dtype = np.float32)
    return fore_sst,flon,flat




####################读取观测数据############################################
def read_obs_data(pmns):
    pys=pmns.year
    pms=pmns.month
    pms = f"{pms:02d}"  # 补零，使月份为两位数
    obs_sst = []
    olat = []
    olon = []
    # %s/MaCOM_daily_%s_t_7day.nc'  % (pmns.strftime('%Y%m%d'),pmns.strftime('%Y%m%d'))
    fname = './obs/%s/%s/%s120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc' %(pys,pms,pmns.strftime('%Y%m%d'))
    #路径中有些没有规律的部分，可以用星号代替，然后用glob补齐路径
    fname_e =glob(fname)
    # print(fname_e)
    if len(fname_e) > 0:
        fpath = fname_e[0]
        # print(fpath)

        if os.path.exists(fpath):
            fid1 = nc.Dataset(fpath ,mode="r" ,format="NETCDF4")
            obs_sst = fid1.variables['analysed_sst'][0,:, :]
            olon=fid1.variables['lon'][:]
            olat=fid1.variables['lat'][:]
            if olat[0] > olat[-1]:  #维度是不是颠倒的，如果颠倒要正过来
                olat = np.flipud(olat)
                obs_sst = np.flipud(obs_sst)
    obs_sst = np.array(obs_sst,dtype = np.float32)
    olat = np.array(olat,dtype = np.float32)
    olon = np.array(olon,dtype = np.float32)            
    return obs_sst,olat,olon

def interp_to_fore(pmns,olat,olon,flat,flon):#传入时间，传入网格化后的预报网格
    #####获取预报数据的经纬度

    
    obs_sst,olon,olat = read_obs_data(pmns) 

    
    fun = interpolate.interp2d(olat, olon,obs_sst, kind='cubic')
    obs_f_sst = fun(flat, flon)
    obs_f_sst = obs_f_sst-273.15  #将温度从开尔文转换成摄氏度
    

    # rlon, rlat = np.meshgrid(rlon, rlat)
    # flon = flon.ravel()
    # flat = flat.ravel()
    # loc0 = np.c_[flon, flat] #拼接行数相同的两个数组  np.r拼接列数相同的两个数组
    # fsst= fore_sst.ravel() #ravel将多维数组降为一维
    # fsst_new = interpolate.griddata(loc0, fsst, (rlon, rlat), method='nearest').reshape(1440, 720)   
                               #loc0插值前的网格，一维拼接，fsst待插值数据，(rlon, rlat)插值后的网格
    return obs_f_sst




#########get series##############################
def rsst_serie():   #预报和观测数据是对齐的
    rsst_series =[]
    time_series = []
    fsst_series = []
    fflon = []
    fflat = []
    temp_pmns = dt.date(2023,7,1)#dt.datetime(2023,9,13)'2023-07-01'
    # temp_pmns = temp_pmns.astype(dt.datetime)
    
    # temp_pmns = temp_pmns.strftime('%Y%m%d')
    fsst,flon,flat =  read_forecast_data(temp_pmns)#读取一个预报数据（lat，lon）用于插值
    # flat = flat.ravel()
    # flon = flon.ravel()
    obs_sst,olon,olat = read_obs_data(temp_pmns) 
    # loc0=np.c_[olon,olat]#拼接行数相同的两个数组  np.r拼接列数相同的两个数组
    for pmns in  np.arange('2023-07-01', '2023-07-08', dtype = np.datetime64):       
        pmns = pmns.astype(dt.datetime)        
        # pmns = pmns.strftime('%Y%m%d')
        obs_f_sst = interp_to_fore(pmns,olat,olon,flat,flon)
        fore_sst,fflon,fflat = read_forecast_data(pmns)


        time_series.append(pmns)
        rsst_series.append(obs_f_sst)
        fsst_series.append(fore_sst)

    rsst_series = np.array(rsst_series,dtype = np.float32)
    time_series = np.array(time_series,dtype = np.datetime64)
    fsst_series = np.array(fsst_series,dtype = np.float32)
    fflat = np.array(fflat,dtype = np.float32)
    fflon = np.array(fflon,dtype = np.float32)
    return time_series,rsst_series,fsst_series,fflat,fflon






def main():

    time_series,rsst_series,fsst_series,flat,flon= rsst_serie()

    fsst=np.array(fsst_series)
    rsst=np.array(rsst_series)
    lat=np.array(flat)
    lon=np.array(flon)
 
    np.save('./data_clean1/fsst_series.npy', fsst)
    np.save('./data_clean1/rsst_series.npy', rsst)
    np.save('./data_clean1/time_series.npy', time_series)

    np.save('./data_clean1/lon_clean.npy', lon)
    np.save('./data_clean1/lat_clean.npy', lat)
# np.save('./data_clean1/mask_clean.npy', mask)

    return


    # np.save('./data_clean_con/time_series.npy', time_series)
    # np.save('./data_clean_con/rsst_series.npy', rsst_series)



    #         # sst_oi_fb.to_csv(fl_csv, header=False, mode='a')
    # return

if __name__ == '__main__':
    main()
                                                                                                                    