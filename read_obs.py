#!usr/bin/python
from heapq import nlargest
import time 
from telnetlib import PRAGMA_HEARTBEAT
import numpy as np
# import numba as nb
import pandas as pd
# import xarray as xr
import os
import netCDF4 as nc
from glob import glob
#forecast data
# filePath = '/work/person/zuzq/EnOI_IAU/OCN_ASSIM/199502/EXP091.pop.h.1995-03.nc'#模式预报数据
# filePath = '/work/person/zuzq/download/OISST/OISSTv2.1/oisst-avhrr-v02r01.19950207.nc'  #观测数据

filePath='./obs/2023/07/20230701120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc'
fileName='20230701'
# filePath = '/Users/liuchunda/Desktop/20230701/20230701120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc'

# filePath = './20230701/20230701120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc'
# print(filePath)
# filePath_e =glob(filePath)
curTime=time.strftime("%Y年%m月%d日%H时%M分%S秒_", time.localtime(time.time()))
np.set_printoptions(threshold=np.inf)
def catchLog(attr,content,content2=None):
    content2 = str(content2)
    with open('./macom_output_data/'+str(curTime)+fileName+'.txt', 'a') as file:
        file.write("\n")
        file.write("\n")

        file.write(attr)
        file.write(':')
        file.write(str(content))
        if content2 != 'None':
            file.write(str(content2))
    return
 
# print(filePath)
if len(filePath) > 0:
    
    fid1 = nc.Dataset(filePath ,mode="r" ,format="NETCDF4")
    #-------------------读取变量属性的具体内容
    variable=fid1.variables['analysed_sst']
    for attr in variable.ncattrs():
        print(f'{attr}:{variable.getncattr(attr)}')
    print('=============================')
    variable2=fid1.variables['time']
    for attr in variable2.ncattrs():
        print(f'{attr}:{variable2.getncattr(attr)}')
    #----------------------------
    # print(fid1.variables.keys())#查看所有变量名
    catchLog('所有变量：',fid1.variables.keys())
    var = fid1.variables['lon']
    # print(var.ncattrs())  # 查看变量的属性
    catchLog('变量属性',var.ncattrs())

    # print(var.dimensions)  # 查看变量的维度
    catchLog('变量维度',var.dimensions)

    # with open('./macom_output_data/'+fileName+'.txt', 'a') as file:
    #     file.write("\n")
    #     file.write('变量维度')
    #     file.write(str(var.dimensions))
    fore_sst = fid1.variables['analysed_sst'][0,:, :]  #('time', 'depth', 'lat', 'lon')  49depth
    
    # print('sst变量维度',fid1.variables['t'].dimensions)
    # fore_sst=np.squeeze
    # (fore_sst)
    # print(fid1['TEMP'])
 
    # print(fore_sst)
    catchLog('analysed_sst',fore_sst)

    # with open('./macom_output_data/'+fileName+'.txt', 'a') as file:
    #     file.write("\n")
    #     file.write("sst")
    #     file.write(str(fore_sst))


    # print(fore_sst.shape)
    flat=  fid1.variables['lat'][:] #二维
    # print(flat.shape)#720
    catchLog('lat',flat.shape,flat)



    # with open('./macom_output_data/'+fileName+'.txt', 'a') as file:
    #     file.write("\n")
    #     file.write("lat")
    #     file.write(str(flat.shape))
    #     file.write(str(flat))
    
    # print(flat)
    flon=  fid1.variables['lon'][:]
    # flon=flon.flatten()#[:,None]
    # print(flon.shape)#1440
    catchLog('lon',flon.shape,flon)
    # depth=  fid1.variables['depth'][:]
    # catchLog('depth',depth.shape,depth)
    

#reanalysed data
     
# filePath = '/work/person/zuzq/download/OISST/OISSTv2.1/oisst-avhrr-v02r01.20130409.nc'
# filePath_e =glob(filePath)
# if len(filePath_e) > 0:
#     fid = nc.Dataset(filePath_e[0J ,mode="r" ,format="NETCDF4")
#     rlon = fid.variables['lon'][:]
#     rlat = fid.variables['lat'][:]
#     dsst = fid.variables['sst'][0,0,:, :]
#     print(rlat.shape)
#     print(rlat)
#     print(rlon.shape)
#     print(rlon)