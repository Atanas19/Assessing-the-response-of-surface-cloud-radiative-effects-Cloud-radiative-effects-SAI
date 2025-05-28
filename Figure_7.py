# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:32:07 2025

@author: ad7gb
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from scipy.stats import ttest_ind

input = 'D:/SAI_Data/UKESM1_Arise_SAI/'
ddirsM = os.listdir(input)

shapefile_path = 'D:/SAI_Data/Africa_Boundaries-shp/Africa_Boundaries.shp'
f = 'D:/SAI_Data/SSP245/LWP/'
ddirsM = os.listdir(f)

###############################################################################
###############################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNS/'
ddirsR = os.listdir(fsai)

SAI_lw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FLNS.203501-206912.nc')
SAI_lw_net.coords['lon'] = (SAI_lw_net.coords['lon'] + 180) % 360 - 180
SAI_lw_net = SAI_lw_net.sortby(SAI_lw_net.lon)

SAI_lw_net = SAI_lw_net.convert_calendar(calendar = 'gregorian', align_on = 'date')

laC = SAI_lw_net.lat.values
loC = SAI_lw_net.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONC, LATC = np.meshgrid(lon,lat)
ds  = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)

laC = ds_clip.lat.values
loC = ds_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONC, LATC = np.meshgrid(lon,lat)
##################### end masking
SAI_lw = ds_clip.values

###############################################################################
###############################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lwR = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwRR = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values 

time = pd.DatetimeIndex(ds['time'].values)
year = time.year.values
month = time.month.values

SAI_lwRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lwRR.shape[1],SAI_lwRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lwRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lwRRR[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lwRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lwRRR[iy-2035,imth-1,:,:] = VV[imth-1]

        
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lwB = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwBB = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values

SAI_lwBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lwBB.shape[1],SAI_lwBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lwBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lwBBB[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lwBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lwBBB[iy-2035,imth-1,:,:] = VV[imth-1]

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lwCA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwCAA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values 

SAI_lwCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lwCAA.shape[1],SAI_lwCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lwCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lwCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lwCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lwCAAA[iy-2035,imth-1,:,:] = VV[imth-1]

###############################################################################
###############################################################################
fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNSC/'
ddirsR = os.listdir(fsai)

SAI_lw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FLNSC.203501-206912.nc')
SAI_lw_net_clr.coords['lon'] = (SAI_lw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_lw_net_clr = SAI_lw_net_clr.sortby(SAI_lw_net_clr.lon)

SAI_lw_net_clr = SAI_lw_net_clr.convert_calendar(calendar = 'gregorian', align_on = 'date')

laC = SAI_lw_net_clr.lat.values
loC = SAI_lw_net_clr.lon.values
lat = laC[np.where((laC>=-10)&(laC <=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
ds  = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SAI_lw_clr = ds_clip.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_clrR = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrRR = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values 

SAI_lw_clrRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lw_clrRR.shape[1],SAI_lw_clrRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lw_clrRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lw_clrRRR[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lw_clrRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lw_clrRRR[iy-2035,imth-1,:,:] = VV[imth-1]
            
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_clrB = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrBB = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values 

SAI_lw_clrBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lw_clrBB.shape[1],SAI_lw_clrBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lw_clrBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lw_clrBBB[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lw_clrBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lw_clrBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_clrCA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrCAA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values

SAI_lw_clrCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_lw_clrCAA.shape[1],SAI_lw_clrCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_lw_clrCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_lw_clrCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SAI_lw_clrCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_lw_clrCAAA[iy-2035,imth-1,:,:] = VV[imth-1]

#################################################################
############################################################################
SAI_CRE_lw = -SAI_lw+SAI_lw_clr
SAI_CRE_lwR = -SAI_lwR+SAI_lw_clrR
SAI_CRE_lwB = -SAI_lwB+SAI_lw_clrB
SAI_CRE_lwCA = -SAI_lwCA+SAI_lw_clrCA


SAI_CRE_lwRR = -SAI_lwRRR+SAI_lw_clrRRR
SAI_CRE_lwBB = -SAI_lwBBB+SAI_lw_clrBBB
SAI_CRE_lwCAA = -SAI_lwCAAA+SAI_lw_clrCAAA

#####################################################################
###########################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNS/'
ddirsR = os.listdir(fsai)

SAI_sw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNS.203501-206912.nc')

SAI_sw_net.coords['lon'] = (SAI_sw_net.coords['lon'] + 180) % 360 - 180
SAI_sw_net = SAI_sw_net.sortby(SAI_sw_net.lon)

laC  = SAI_sw_net.lat.values
loC  = SAI_sw_net.lon.values
lat = laC[np.where((laC>=-10)& (laC <=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'), lat = lat, lon = lon)['FSNS']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SAI_sw = ds_clip.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_swR = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swRR = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SAI_swRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_swRR.shape[1],SAI_swRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_swRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_swRRR[iy-2035,imth,:,:] = np.nan
            else:
                SAI_swRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_swRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_swB = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swBB = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SAI_swBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_swBB.shape[1],SAI_swBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_swBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_swBBB[iy-2035,imth,:,:] = np.nan
            else:
                SAI_swBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_swBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_swCA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swCAA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SAI_swCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_swCAA.shape[1],SAI_swCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_swCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_swCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SAI_swCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_swCAAA[iy-2035,imth-1,:,:] = VV[imth-1]

####################################################################################
#######################################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNSC/'
ddirsR = os.listdir(fsai)

SAI_sw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNSC.203501-206912.nc')

SAI_sw_net_clr.coords['lon'] = (SAI_sw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_sw_net_clr = SAI_sw_net_clr.sortby(SAI_sw_net_clr.lon)

laC  = SAI_sw_net_clr.lat.values
loC  = SAI_sw_net_clr.lon.values

lat = laC[np.where((laC>=-10)& (laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SAI_sw_clr = ds_clip.values


####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_clrR = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrRR = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 

SAI_sw_clrRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_sw_clrRR.shape[1],SAI_sw_clrRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_sw_clrRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_sw_clrRRR[iy-2035,imth,:,:] = np.nan
            else:
                SAI_sw_clrRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_sw_clrRRR[iy-2035,imth-1,:,:] = VV[imth-1]
            
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_clrB = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrBB = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 


SAI_sw_clrBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_sw_clrBB.shape[1],SAI_sw_clrBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_sw_clrBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_sw_clrBBB[iy-2035,imth,:,:] = np.nan
            else:
                SAI_sw_clrBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_sw_clrBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_clrCA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrCAA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 


SAI_sw_clrCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_sw_clrCAA.shape[1],SAI_sw_clrCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_sw_clrCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_sw_clrCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SAI_sw_clrCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_sw_clrCAAA[iy-2035,imth-1,:,:] = VV[imth-1]
            
##################################################
################################################

SAI_CRE_sw = SAI_sw-SAI_sw_clr
SAI_CRE_swR = SAI_swR-SAI_sw_clrR
SAI_CRE_swB = SAI_swB-SAI_sw_clrB
SAI_CRE_swCA = SAI_swCA-SAI_sw_clrCA

SAI_CRE = SAI_CRE_sw+SAI_CRE_lw
SAI_CRE_R = SAI_CRE_swR+SAI_CRE_lwR
SAI_CRE_B = SAI_CRE_swB+SAI_CRE_lwB
SAI_CRE_CA = SAI_CRE_swCA+SAI_CRE_lwCA



SAI_CRE_swRR = SAI_swRRR-SAI_sw_clrRRR
SAI_CRE_swBB = SAI_swBBB-SAI_sw_clrBBB
SAI_CRE_swCAA = SAI_swCAAA-SAI_sw_clrCAAA

SAI_CRE_RR = SAI_CRE_swRR+SAI_CRE_lwRR
SAI_CRE_BB = SAI_CRE_swBB+SAI_CRE_lwBB
SAI_CRE_CAA = SAI_CRE_swCAA+SAI_CRE_lwCAA


#####################################################################
#####################################################################
########################## precipitation data #######################
 
fsai = 'D:/SAI_Data/ARISE_WACCM/PRECT/'
ddirs  = os.listdir(fsai)

SAI_pr = xr.open_mfdataset(fsai+'Ensmean_b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.PRECT.203501-206912.nc')

SAI_pr.coords['lon'] = (SAI_pr.coords['lon'] + 180) % 360 - 180
SAI_pr = SAI_pr.sortby(SAI_pr.lon)

SAI_pr =1000*3600*24*SAI_pr
laC  = SAI_pr.lat.values
loC  = SAI_pr.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['PRECT']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)

laC  = ds_clip.lat.values
loC  = ds_clip.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
LONMod, LATMod = np.meshgrid(lon,lat)

##################### end masking
SAI_prr = ds_clip.values

####################################################################
########################################################################
##################### red box

latwa = laC[np.where((laC>=7)&(laC<=13))[0]]
lonwa = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_prR = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latwa, lon = lonwa).groupby('time.month').mean(dim ='time')['PRECT'].values 
SAI_prRR = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latwa, lon = lonwa)['PRECT'].values 

SAI_prRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_prRR.shape[1],SAI_prRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_prRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_prRRR[iy-2035,imth,:,:] = np.nan
            else:
                SAI_prRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_prRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
latsa = laC[np.where((laC>=16)&(laC<=22))[0]]
lonsa= loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_prB = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latsa, lon = lonsa).groupby('time.month').mean(dim ='time')['PRECT'].values 
SAI_prBB = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latsa, lon = lonsa)['PRECT'].values 

SAI_prBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_prBB.shape[1],SAI_prBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_prBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_prBBB[iy-2035,imth,:,:] = np.nan
            else:
                SAI_prBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_prBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
latca = laC[np.where((laC>=-2)&(laC<=5))[0]]
lonca = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_prCA = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latca, lon = lonca).groupby('time.month').mean(dim ='time')['PRECT'].values 
SAI_prCAA = SAI_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latca, lon = lonca)['PRECT'].values 


SAI_prCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SAI_prCAA.shape[1],SAI_prCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SAI_prCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SAI_prCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SAI_prCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SAI_prCAAA[iy-2035,imth-1,:,:] = VV[imth-1]

###############################################################################
###############################################################################
###############################################################################

LatTime_SAI_prR  = np.nanmean(SAI_prR, axis=2)
LatTime_SAI_prB  = np.nanmean(SAI_prB, axis=2)
LatTime_SAI_prCA = np.nanmean(SAI_prCA, axis=2)

LatTime_CRE_SAI_swR  =   np.nanmean(SAI_CRE_swR , axis=2)  
LatTime_CRE_SAI_swB  =   np.nanmean(SAI_CRE_swB , axis=2)
LatTime_CRE_SAI_swCA =   np.nanmean(SAI_CRE_swCA , axis=2)

LatTime_CRE_SAI_lwR  =   np.nanmean(SAI_CRE_lwR , axis=2)  
LatTime_CRE_SAI_lwB  =   np.nanmean(SAI_CRE_lwB , axis=2)
LatTime_CRE_SAI_lwCA =   np.nanmean(SAI_CRE_lwCA , axis=2)



LatTime_SAI_prRR  = np.nanmean(SAI_prRRR, axis=3)
LatTime_SAI_prBB  = np.nanmean(SAI_prBBB, axis=3)
LatTime_SAI_prCAA = np.nanmean(SAI_prCAAA, axis=3)

LatTime_CRE_SAI_swRR  =   np.nanmean(SAI_CRE_swRR , axis=3)  
LatTime_CRE_SAI_swBB  =   np.nanmean(SAI_CRE_swBB , axis=3)
LatTime_CRE_SAI_swCAA =   np.nanmean(SAI_CRE_swCAA , axis=3)

LatTime_CRE_SAI_lwRR  =   np.nanmean(SAI_CRE_lwRR , axis=3)  
LatTime_CRE_SAI_lwBB  =   np.nanmean(SAI_CRE_lwBB , axis=3)
LatTime_CRE_SAI_lwCAA =   np.nanmean(SAI_CRE_lwCAA , axis=3)

###############################################################################
###############################################################################
############################### ssp245 
####################################################################################
#######################################################################################



fssp   = 'D:/SAI_Data/SSP245/FLNS/'
ddirs  = os.listdir(fssp)

SSP245_lw_net = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FLNS.201501-206912.nc')
SSP245_lw_net.coords['lon'] = (SSP245_lw_net.coords['lon'] + 180) % 360 - 180
SSP245_lw_net = SSP245_lw_net.sortby(SSP245_lw_net.lon)

SSP245_lw_net = SSP245_lw_net.convert_calendar(calendar = 'gregorian', align_on = 'date')

laC = SSP245_lw_net.lat.values
loC = SSP245_lw_net.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
LONC, LATC = np.meshgrid(lon,lat)
ds  = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SSP245_lw = ds_clip.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lwR = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwRR = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values 

SSP245_lwRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lwRR.shape[1],SSP245_lwRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lwRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lwRRR[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lwRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lwRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lwB = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwBB = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values 

SSP245_lwBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lwBB.shape[1],SSP245_lwBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lwBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lwBBB[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lwBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lwBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lwCA = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwCAA = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].values 

SSP245_lwCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lwCAA.shape[1],SSP245_lwCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lwCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lwCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lwCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lwCAAA[iy-2035,imth-1,:,:] = VV[imth-1]
###############################################################################
###############################################################################
fssp   = 'D:/SAI_Data/SSP245/FLNSC/'
ddirs  = os.listdir(fssp)

SSP245_lw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FLNSC.201501-206912.nc')
SSP245_lw_net_clr.coords['lon'] = (SSP245_lw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_lw_net_clr = SSP245_lw_net_clr.sortby(SSP245_lw_net_clr.lon)

SSP245_lw_net_clr = SSP245_lw_net_clr.convert_calendar(calendar = 'gregorian', align_on = 'date')

laC = SSP245_lw_net_clr.lat.values
loC = SSP245_lw_net_clr.lon.values
lat = laC[np.where((laC>=-10)&(laC <=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

ds  = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SSP245_lw_clr = ds_clip.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_clrR = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrRR = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values 

SSP245_lw_clrRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lw_clrRR.shape[1],SSP245_lw_clrRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lw_clrRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lw_clrRRR[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lw_clrRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lw_clrRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_clrB = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrBB = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values 

SSP245_lw_clrBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lw_clrBB.shape[1],SSP245_lw_clrBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lw_clrBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lw_clrBBB[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lw_clrBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lw_clrBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_clrCA = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrCAA = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].values 

SSP245_lw_clrCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_lw_clrCAA.shape[1],SSP245_lw_clrCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_lw_clrCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_lw_clrCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_lw_clrCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_lw_clrCAAA[iy-2035,imth-1,:,:] = VV[imth-1]
###############################################################
##########################################################################
SSP245_CRE_lw   = -SSP245_lw +SSP245_lw_clr
SSP245_CRE_lwR  = -SSP245_lwR +SSP245_lw_clrR
SSP245_CRE_lwB  = -SSP245_lwB +SSP245_lw_clrB
SSP245_CRE_lwCA = -SSP245_lwCA +SSP245_lw_clrCA


SSP245_CRE_lwRR  = -SSP245_lwRRR +SSP245_lw_clrRRR
SSP245_CRE_lwBB  = -SSP245_lwBBB +SSP245_lw_clrBBB
SSP245_CRE_lwCAA = -SSP245_lwCAAA +SSP245_lw_clrCAAA


#####################################################################
###########################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNS/'
ddirs  = os.listdir(fssp)

SSP245_sw_net = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FSNS.201501-206912.nc')

SSP245_sw_net.coords['lon'] = (SSP245_sw_net.coords['lon'] + 180) % 360 - 180
SSP245_sw_net = SSP245_sw_net.sortby(SSP245_sw_net.lon)

laC  = SSP245_sw_net.lat.values
loC  = SSP245_sw_net.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)
##################### end masking
SSP245_sw = ds_clip.values


####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_swR = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swRR = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SSP245_swRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_swRR.shape[1],SSP245_swRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_swRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_swRRR[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_swRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_swRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_swB = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swBB = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SSP245_swBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_swBB.shape[1],SSP245_swBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_swBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_swBBB[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_swBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_swBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_swCA = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swCAA = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].values 

SSP245_swCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_swCAA.shape[1],SSP245_swCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_swCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_swCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_swCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_swCAAA[iy-2035,imth-1,:,:] = VV[imth-1]
####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNSC/'
ddirs  = os.listdir(fssp)

SSP245_sw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FSNSC.201501-206912.nc')

SSP245_sw_net_clr.coords['lon'] = (SSP245_sw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_sw_net_clr = SSP245_sw_net_clr.sortby(SSP245_sw_net_clr.lon)

laC  = SSP245_sw_net_clr.lat.values
loC  = SSP245_sw_net_clr.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)

laC  = ds_clip.lat.values
loC  = ds_clip.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
LONMod, LATMod = np.meshgrid(lon,lat)

##################### end masking
SSP245_sw_clr = ds_clip.values


####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_clrR = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrRR = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 

SSP245_sw_clrRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_sw_clrRR.shape[1],SSP245_sw_clrRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_sw_clrRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_sw_clrRRR[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_sw_clrRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_sw_clrRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_clrB = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrBB = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 

SSP245_sw_clrBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_sw_clrBB.shape[1],SSP245_sw_clrBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_sw_clrBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_sw_clrBBB[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_sw_clrBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_sw_clrBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_clrCA = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrCAA = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].values 

SSP245_sw_clrCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_sw_clrCAA.shape[1],SSP245_sw_clrCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_sw_clrCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_sw_clrCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_sw_clrCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_sw_clrCAAA[iy-2035,imth-1,:,:] = VV[imth-1]

SSP245_CRE_sw   = SSP245_sw- SSP245_sw_clr
SSP245_CRE_swR  = SSP245_swR- SSP245_sw_clrR
SSP245_CRE_swB  = SSP245_swB- SSP245_sw_clrB
SSP245_CRE_swCA = SSP245_swCA- SSP245_sw_clrCA

SSP245_CRE    = SSP245_CRE_sw + SSP245_CRE_lw
SSP245_CRE_R  = SSP245_CRE_swR + SSP245_CRE_lwR
SSP245_CRE_B  = SSP245_CRE_swB + SSP245_CRE_lwB
SSP245_CRE_CA = SSP245_CRE_swCA + SSP245_CRE_lwCA



SSP245_CRE_swRR  = SSP245_swRRR- SSP245_sw_clrRRR
SSP245_CRE_swBB  = SSP245_swBBB- SSP245_sw_clrBBB
SSP245_CRE_swCAA = SSP245_swCAAA- SSP245_sw_clrCAAA

SSP245_CRE_RR  = SSP245_CRE_swRR + SSP245_CRE_lwRR
SSP245_CRE_BB  = SSP245_CRE_swBB + SSP245_CRE_lwBB
SSP245_CRE_CAA = SSP245_CRE_swCAA + SSP245_CRE_lwCAA

#####################################################################
#####################################################################
########################## precipitation data #######################
###############################################################################
###############################################################################
fssp = 'D:/SAI_Data/SSP245/PRECT/'
ddirs  = os.listdir(fssp)

SSP245_pr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.PRECT.201501-206912.nc')

SSP245_pr.coords['lon'] = (SSP245_pr.coords['lon'] + 180) % 360 - 180
SSP245_pr = SSP245_pr.sortby(SSP245_pr.lon)

SSP245_pr =1000*3600*24*SSP245_pr
laC  = SSP245_pr.lat.values
loC  = SSP245_pr.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)
ds  = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['PRECT']

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
ds = ds.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(ds.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
ds_clip = ds.rio.clip(geometries, mskk.crs)

laC  = ds_clip.lat.values
loC  = ds_clip.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
LONMod, LATMod = np.meshgrid(lon,lat)
##################### end masking
SSP245_prr = ds_clip.values

####################################################################
########################################################################
##################### red box

latwa = laC[np.where((laC>=7)&(laC<=13))[0]]
lonwa = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_prR = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latwa, lon = lonwa).groupby('time.month').mean(dim ='time')['PRECT'].values 
SSP245_prRR = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latwa, lon = lonwa)['PRECT'].values 

SSP245_prRRR = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_prRR.shape[1],SSP245_prRR.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_prRR[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_prRRR[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_prRRR[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_prRRR[iy-2035,imth-1,:,:] = VV[imth-1]
#################### black box 
latsa = laC[np.where((laC>=16)&(laC<=22))[0]]
lonsa = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_prB = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latsa, lon = lonsa).groupby('time.month').mean(dim ='time')['PRECT'].values 
SSP245_prBB = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latsa, lon = lonsa)['PRECT'].values 

SSP245_prBBB = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_prBB.shape[1],SSP245_prBB.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_prBB[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_prBBB[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_prBBB[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_prBBB[iy-2035,imth-1,:,:] = VV[imth-1]
#################### Green box 
latca = laC[np.where((laC>=-2)&(laC<=5))[0]]
lonca = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_prCA = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latca, lon = lonca).groupby('time.month').mean(dim ='time')['PRECT'].values 
SSP245_prCAA = SSP245_pr.sel(time = slice('2035-02-01','2069-12-31'),lat = latca, lon = lonca)['PRECT'].values 

SSP245_prCAAA = np.nan*np.zeros([len(np.unique(year)), len(np.unique(month)), SSP245_prCAA.shape[1],SSP245_prCAA.shape[2]])
for iy in np.unique(year):
    YY = np.where(year==iy)
    VV = SSP245_prCAA[YY]
    if iy==2035:
        MM = np.arange(0,12)
        for imth in MM:
            if imth==0:
                SSP245_prCAAA[iy-2035,imth,:,:] = np.nan
            else:
                SSP245_prCAAA[iy-2035,imth,:,:] = VV[imth-1]
    else:
        MM = np.unique(month)
        for imth in MM:
            SSP245_prCAAA[iy-2035,imth-1,:,:] = VV[imth-1]
###############################################################################
###############################################################################
###############################################################################

LatTime_ssp245_prR  = np.nanmean(SSP245_prR, axis=2)
LatTime_ssp245_prB  = np.nanmean(SSP245_prB, axis=2)
LatTime_ssp245_prCA = np.nanmean(SSP245_prCA, axis=2)

LatTime_CRE_ssp245_swR  =   np.nanmean(SSP245_CRE_swR , axis=2)  
LatTime_CRE_ssp245_swB  =   np.nanmean(SSP245_CRE_swB , axis=2)
LatTime_CRE_ssp245_swCA =   np.nanmean(SSP245_CRE_swCA , axis=2)

LatTime_CRE_ssp245_lwR  =   np.nanmean(SSP245_CRE_lwR , axis=2)  
LatTime_CRE_ssp245_lwB  =   np.nanmean(SSP245_CRE_lwB , axis=2)
LatTime_CRE_ssp245_lwCA =   np.nanmean(SSP245_CRE_lwCA , axis=2)


LatTime_ssp245_prRR  = np.nanmean(SSP245_prRRR, axis=3)
LatTime_ssp245_prBB  = np.nanmean(SSP245_prBBB, axis=3)
LatTime_ssp245_prCAA = np.nanmean(SSP245_prCAAA, axis=3)

LatTime_CRE_ssp245_swRR  =   np.nanmean(SSP245_CRE_swRR , axis=3)  
LatTime_CRE_ssp245_swBB  =   np.nanmean(SSP245_CRE_swBB , axis=3)
LatTime_CRE_ssp245_swCAA =   np.nanmean(SSP245_CRE_swCAA , axis=3)

LatTime_CRE_ssp245_lwRR  =   np.nanmean(SSP245_CRE_lwRR , axis=3)  
LatTime_CRE_ssp245_lwBB  =   np.nanmean(SSP245_CRE_lwBB , axis=3)
LatTime_CRE_ssp245_lwCAA =   np.nanmean(SSP245_CRE_lwCAA , axis=3)


###############################################################################
#################################### Biases ###################################
###############################################################################

Bias_latTime_CRE_swR  = LatTime_CRE_SAI_swR  -LatTime_CRE_ssp245_swR
Bias_latTime_CRE_swB  = LatTime_CRE_SAI_swB  -LatTime_CRE_ssp245_swB
Bias_latTime_CRE_swCA = LatTime_CRE_SAI_swCA -LatTime_CRE_ssp245_swCA

Bias_latTime_CRE_lwR  = LatTime_CRE_SAI_lwR  -LatTime_CRE_ssp245_lwR
Bias_latTime_CRE_lwB  = LatTime_CRE_SAI_lwB  -LatTime_CRE_ssp245_lwB
Bias_latTime_CRE_lwCA = LatTime_CRE_SAI_lwCA -LatTime_CRE_ssp245_lwCA

Bias_latTime_prR  = LatTime_SAI_prR  -LatTime_ssp245_prR
Bias_latTime_prB  = LatTime_SAI_prB  -LatTime_ssp245_prB
Bias_latTime_prCA = LatTime_SAI_prCA -LatTime_ssp245_prCA


###############################################################################
#################################################################################

SAII_R = [LatTime_CRE_SAI_swRR, LatTime_CRE_SAI_lwRR, LatTime_SAI_prRR]
SAII_B = [LatTime_CRE_SAI_swBB, LatTime_CRE_SAI_lwBB, LatTime_SAI_prBB]
SAII_CA = [LatTime_CRE_SAI_swCAA, LatTime_CRE_SAI_lwCAA, LatTime_SAI_prCAA]

SSP245_R = [LatTime_CRE_ssp245_swRR, LatTime_CRE_ssp245_lwRR, LatTime_ssp245_prRR]
SSP245_B = [LatTime_CRE_ssp245_swBB, LatTime_CRE_ssp245_lwBB, LatTime_ssp245_prBB]
SSP245_CA = [LatTime_CRE_ssp245_swCAA, LatTime_CRE_ssp245_lwCAA, LatTime_ssp245_prCAA]


PVAL_R = [];PVAL_B = [];PVAL_CA = []
for i in np.arange(0 ,3): #ttest_rel
    statres, pval1 = ttest_ind(SAII_R[i], SSP245_R[i], axis=0, nan_policy='propagate', alternative='two-sided')
    spy1 = np.where(pval1 < 0.05 , 1,np.nan)
    PVAL_R.append(spy1)
    
    statres, pval1 = ttest_ind(SAII_B[i], SSP245_B[i], axis=0, nan_policy='propagate', alternative='two-sided')
    spy1 = np.where(pval1 < 0.05 , 1,np.nan)
    PVAL_B.append(spy1)
    
    statres, pval1 = ttest_ind(SAII_CA[i], SSP245_CA[i], axis=0, nan_policy='propagate', alternative='two-sided')
    spy1 = np.where(pval1 < 0.05 , 1,np.nan)
    PVAL_CA.append(spy1)

PVAL_R = np.array(PVAL_R)
PVAL_B = np.array(PVAL_B)
PVAL_CA = np.array(PVAL_CA)
#######################################################################
######################################################################
##############################################################


IND = ['a) SWA    ΔSWCRE','b) SWA     ΔLWCRE','c) SWA      Δpr'] #ΔNet

clevv = np.arange(-3.6,3.8,0.2)
levels = np.arange(-126, -4)
time = np.arange(0,12)
MTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
cmap = plt.get_cmap('bwr')
cmap1 = plt.get_cmap('bwr_r')
cmap_r = plt.get_cmap('PuOr')

fig = plt.figure(figsize=(28,10))

gs = fig.add_gridspec(3, 3, bottom=0.0, top=0.85, left=0.05, 
                      right=0.5, wspace=0.22, hspace=0.30)

ax0 = fig.add_subplot(gs[0, 0]) # ,projection = ccrs.PlateCarree()
cs=ax0.contourf(time,latwa, Bias_latTime_CRE_swR.T,clevv,
      cmap= cmap ,extend='both') #,transform = ccrs.PlateCarree()

css =ax0.contour(time,latwa, LatTime_CRE_ssp245_swR.T, levels[::4],
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax0.clabel(css, css.levels[::4], fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latwa)
ax0.scatter(TT.T*PVAL_R[0], LA.T*PVAL_R[0], s=10, marker='.', color = 'black') 

ax0.set_title(IND[0],fontweight = 'bold',fontsize=15)
ax0.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax0.set_xticklabels(MTH, rotation =45,  fontsize=12)

# Define the yticks for latitude
ax0.set_yticks(np.arange(7,13,0.5), )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax0.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax0.axes.get_xaxis().set_ticklabels
ax0.axes.get_yaxis().set_ticklabels
ax0.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=15,fontweight = 'bold')

##############################################################################
###########################################################################
clevv = np.arange(-2,2.2,.2)
levels = np.arange(0, 21)

ax1 = fig.add_subplot(gs[0, 1])#,Δprojection=ccrs.PlateCarree()
cs=ax1.contourf(time,latwa, Bias_latTime_CRE_lwR.T, clevv,cmap= cmap ,extend='both') #,transform = ccrs.PlateCarree()

css =ax1.contour(time,latwa, LatTime_CRE_ssp245_lwR.T, levels[::2],
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax1.clabel(css, css.levels[::2], fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f")  

TT, LA = np.meshgrid(time, latwa)
ax1.scatter(TT.T*PVAL_R[1], LA.T*PVAL_R[1], s=10, marker='.', color = 'black') 

#ax1.coastlines(linewidth=4)
ax1.set_title(IND[1],fontweight = 'bold',fontsize=15)
ax1.set_xticks(np.arange(0,12,1))#, crs=ccrs.PlateCarree()

ax1.set_yticks(np.arange(7,13,0.5), )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax1.axes.get_xaxis().set_ticklabels
ax1.axes.get_yaxis().set_ticklabels
ax1.axes.axis('tight')
ax1.set_yticklabels([])
#cbar_ax = fig.add_axes([0.51, 0.69, 0.006, 0.15]) 
cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=15,fontweight = 'bold')
ax1.set_xticklabels(MTH, rotation =45,  fontsize=12)

##############################################################################
###########################################################################
clevv = np.arange(-0.8,0.9,0.1)
levels = np.arange(0,13)

ax2 = fig.add_subplot(gs[0, 2]) #projection=ccrs.PlateCarree()
cs=ax2.contourf(time, latwa, Bias_latTime_prR.T,clevv, cmap= cmap1 ,extend='both') # transform = ccrs.PlateCarree(),

css =ax2.contour(time, latwa, LatTime_ssp245_prR.T, levels,
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax2.clabel(css, levels, fontsize = 15, inline =True, colors='k',fmt = "%0.0f")   
ax2.scatter(TT.T*PVAL_R[2], LA.T*PVAL_R[2], s=10, marker='.', color = 'black') 
#ax2.coastlines(linewidth=4)
ax2.set_title(IND[2],fontweight = 'bold',fontsize=15)
ax2.set_xticks(np.arange(0,12,1)) #, crs=ccrs.PlateCarree()

ax2.set_yticks(np.arange(7,13,0.5), )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax2.axes.get_xaxis().set_ticklabels
ax2.axes.get_yaxis().set_ticklabels
ax2.axes.axis('tight')
ax2.set_yticklabels([])
cbar=fig.colorbar(cs, orientation='vertical',extendrect = 'True')
cbar.set_label('mm/day', fontsize=15,fontweight = 'bold')
ax2.set_xticklabels(MTH, rotation =45,  fontsize=12)
###################################################################
#################################################################
################# sahara region

IND = ['d) SAH    SWCRE','e) SAH     LWCRE','f) SAH   pr'] 
clevv = np.arange(-1.8,2,0.2)
levels = np.arange(-28, 0)

ax3 = fig.add_subplot(gs[1, 0])
cs=ax3.contourf(time,latsa, Bias_latTime_CRE_swB.T, clevv,cmap= cmap ,extend='both') 
css =ax3.contour(time,latsa, LatTime_CRE_ssp245_swB.T, levels[::2],
                  linewidths = 1,colors='gray', extend='both') 
ax3.clabel(css, css.levels[::2], fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latsa)
ax3.scatter(TT.T*PVAL_B[0], LA.T*PVAL_B[0], s=10, marker='.', color = 'black')

ax3.set_title(IND[0],fontweight = 'bold',fontsize=15)
ax3.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax3.set_xticklabels(MTH, rotation =45,  fontsize=12)
# Define the yticks for latitude
ax3.set_yticks(np.arange(16,22,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax3.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax3.axes.get_xaxis().set_ticklabels
ax3.axes.get_yaxis().set_ticklabels
ax3.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=15,fontweight = 'bold')

###################################################################
#################################################################

clevv = np.arange(-1,1.2,0.2)
levels = np.arange(0,7)

ax4 = fig.add_subplot(gs[1, 1])
cs=ax4.contourf(time,latsa, Bias_latTime_CRE_lwB.T, clevv,cmap= cmap ,extend='both') 
css =ax4.contour(time,latsa, LatTime_CRE_ssp245_lwB.T, levels,
                  linewidths = 1,colors='gray', extend='both') 
ax4.clabel(css, css.levels, fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latsa)
ax4.scatter(TT.T*PVAL_B[1], LA.T*PVAL_B[1], s=10, marker='.', color = 'black') 

ax4.set_title(IND[1],fontweight = 'bold',fontsize=15)
ax4.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax4.set_xticklabels(MTH, rotation =45,  fontsize=12)

ax4.set_yticks(np.arange(16,22,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax4.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax4.axes.get_xaxis().set_ticklabels
ax4.axes.get_yaxis().set_ticklabels
ax4.set_yticklabels([])
ax4.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=20,fontweight = 'bold')

###################################################################
#################################################################

clevv = np.arange(-0.1,0.101,0.01)
levels = np.arange(0,2,0.1)


ax5 = fig.add_subplot(gs[1, 2]) #projection=ccrs.PlateCarree()
cs=ax5.contourf(time, latsa, Bias_latTime_prB.T,clevv, cmap= cmap1 ,extend='both') # transform = ccrs.PlateCarree(),
css =ax5.contour(time, latsa, LatTime_ssp245_prB.T, levels,linewidths = 1,colors='gray', extend='both') 
ax5.clabel(css, levels, fontsize = 15, inline =True, colors='k',fmt = "%0.0f")   
ax5.scatter(TT.T*PVAL_B[2], LA.T*PVAL_B[2], s=10, marker='.', color = 'black')
#ax2.coastlines(linewidth=4)
ax5.set_title(IND[2],fontweight = 'bold',fontsize=15)
ax5.set_xticks(np.arange(0,12,1)) #, crs=ccrs.PlateCarree()
ax5.set_xticklabels(MTH, rotation =45,  fontsize=12)

lat_formatter = cticker.LatitudeFormatter()
ax5.yaxis.set_major_formatter(lat_formatter)

ax5.set_yticks(np.arange(16,22,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax5.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax5.axes.get_xaxis().set_ticklabels
ax5.axes.get_yaxis().set_ticklabels
ax5.set_yticklabels([])
ax5.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('mm/day', fontsize=15,fontweight = 'bold')

#############################################################
#############################################################
################### Central Africa 
IND = ['g) CA    SWCRE','h) CA     LWCRE','i) CA   pr'] 
clevv = np.arange(-2.8,3,0.2)
levels = np.arange(-106,-16)

ax6 = fig.add_subplot(gs[2, 0])

cs=ax6.contourf(time,latca, Bias_latTime_CRE_swCA.T, clevv,cmap= cmap ,extend='both') 
css =ax6.contour(time,latca, LatTime_CRE_ssp245_swCA.T, levels[::3],
                  linewidths = 1,colors='gray', extend='both') 
ax6.clabel(css, css.levels[::3], fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latca)
ax6.scatter(TT.T*PVAL_CA[0], LA.T*PVAL_CA[0], s=10, marker='.', color = 'black')

ax6.set_title(IND[0],fontweight = 'bold',fontsize=15)
ax6.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax6.set_xticklabels(MTH, rotation =45,  fontsize=12)
# Define the yticks for latitude
ax6.set_yticks(np.arange(-2,5,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax6.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax6.axes.get_xaxis().set_ticklabels
ax6.axes.get_yaxis().set_ticklabels
ax6.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=15,fontweight = 'bold')


#########################################################################
###################################################################
clevv = np.arange(-1.6,1.8,0.2)
levels = np.arange(0,20)

ax7 = fig.add_subplot(gs[2, 1])

cs=ax7.contourf(time,latca, Bias_latTime_CRE_lwCA.T,clevv, cmap= cmap ,extend='both') 
css =ax7.contour(time,latca, LatTime_CRE_ssp245_lwCA.T, levels,
                  linewidths = 1,colors='gray', extend='both') 
ax7.clabel(css, css.levels, fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latca)
ax7.scatter(TT.T*PVAL_CA[1], LA.T*PVAL_CA[1], s=10, marker='.', color = 'black')

ax7.set_title(IND[1],fontweight = 'bold',fontsize=15)
ax7.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax7.set_xticklabels(MTH, rotation =45,  fontsize=12)

ax7.set_yticks(np.arange(-2,5,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax7.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax7.axes.get_xaxis().set_ticklabels
ax7.axes.get_yaxis().set_ticklabels
ax7.set_yticklabels([])
ax7.axes.axis('tight')

cbar=fig.colorbar(cs, orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=15,fontweight = 'bold')

#########################################################################
###################################################################

clevv = np.arange(-0.9,1,0.1)
levels = np.arange(0, 10)

ax8 = fig.add_subplot(gs[2, 2])

TT,LA = np.meshgrid(time,latca)
cs=ax8.contourf(time,latca, Bias_latTime_prCA.T,clevv, cmap= cmap1 ,extend='both') 
css =ax8.contour(time,latca, LatTime_ssp245_prCA.T, levels,
                  linewidths = 1,colors='gray', extend='both') 
ax8.clabel(css, css.levels, fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 

TT, LA = np.meshgrid(time, latca)
ax8.scatter(TT.T*PVAL_CA[2], LA.T*PVAL_CA[2], s=10, marker='.', color = 'black')


ax8.set_title(IND[2],fontweight = 'bold',fontsize=15)
ax8.set_xticks(np.arange(0,12,1),)#crs=ccrs.PlateCarree()
ax8.set_xticklabels(MTH, rotation =45,  fontsize=12)

ax8.set_yticks(np.arange(-2,5,0.5) )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax8.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax8.axes.get_xaxis().set_ticklabels
ax8.axes.get_yaxis().set_ticklabels
ax8.set_yticklabels([])
ax8.axes.axis('tight')

cbar=fig.colorbar(cs,  orientation='vertical',extendrect = 'True')
cbar.set_label('mm/day', fontsize=15,fontweight = 'bold')


plt.savefig("D:/SAI_Data/Figure_7.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_7.pdf", dpi=700, bbox_inches='tight')


