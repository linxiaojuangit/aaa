#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt  
import numpy as np
import netCDF4 as nc

# 读取nc文件
file_path = './macom_data/20230701/MaCOM_daily_20230701_t_7day.nc'
fid = nc.Dataset(file_path,mode="r" ,format="NETCDF4")
sst = fid.variables['t'][0,0,:, :]
lat = fid.variables['lat'][:]
lon = fid.variables['lon'][:]
fid.close()

# 绘图
plt.figure(figsize=(10, 5))

plt.contourf(lon, lat, sst, cmap='jet')
plt.colorbar()
plt.title('SST')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



