# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:43:54 2024

@author: ad7gb
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter
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


SAI_CRE_lw = []; SAI_CRE_lwR = [];
SAI_CRE_lwB = []; SAI_CRE_lwCA = [];

fileN = [1,2,3,4,5,6,7,8,9,10]

for ifile in fileN:
    print(ifile)
    fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNS/'
    if ifile<=9:
        SAI_lw_net = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00'+str(ifile)+'.cam.h0.FLNS.203501-206912.nc')
    else:
        SAI_lw_net = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0'+str(ifile)+'.cam.h0.FLNS.203501-206912.nc')
    
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
    SAI_lwR = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),
                             lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SAI_lwB = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),
                             lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SAI_lwCA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),
                              lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
        
    ###############################################################################
    ###############################################################################
    fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNSC/'
    ddirsR = os.listdir(fsai)
    if ifile<=9:
        SAI_lw_net_clr = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00'+str(ifile)+'.cam.h0.FLNSC.203501-206912.nc')
    else:
        SAI_lw_net_clr = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0'+str(ifile)+'.cam.h0.FLNSC.203501-206912.nc')
        
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
    SAI_lw_clrR = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                     lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SAI_lw_clrB = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                     lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SAI_lw_clrCA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                      lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 
    
    #################################################################
    ############################################################################
    SAI_CRE_lw.append(-SAI_lw+SAI_lw_clr)
    SAI_CRE_lwR.append(-SAI_lwR+SAI_lw_clrR)
    SAI_CRE_lwB.append(-SAI_lwB+SAI_lw_clrB)
    SAI_CRE_lwCA.append(-SAI_lwCA+SAI_lw_clrCA)

SAI_CRE_lw = np.nanmean(np.array(SAI_CRE_lw), axis=0)
SAI_CRE_lwR = np.array(SAI_CRE_lwR)
SAI_CRE_lwB= np.array(SAI_CRE_lwB)
SAI_CRE_lwCA= np.array(SAI_CRE_lwCA)

#####################################################################
###########################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNS/'
ddirsR = os.listdir(fsai)

fileN = [1,2,3,4,5,6,7,8,9,10]

SAI_CRE_sw = [];SAI_CRE_swR = [];
SAI_CRE_swB = []; SAI_CRE_swCA = []

for ifile in fileN:
    print(ifile)
    fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNS/'
    if ifile<=9:
        SAI_sw_net = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00'+str(ifile)+'.cam.h0.FSNS.203501-206912.nc')
    else:
        SAI_sw_net = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0'+str(ifile)+'.cam.h0.FSNS.203501-206912.nc')

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
    SAI_swR = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),
                             lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SAI_swB = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),
                             lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SAI_swCA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),
                              lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 
       
    ####################################################################################
    #######################################################################################
    
    fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNSC/'
    ddirsR = os.listdir(fsai)

    if ifile<=9:
        SAI_sw_net_clr = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.00'+str(ifile)+'.cam.h0.FSNSC.203501-206912.nc')
    else:
        SAI_sw_net_clr = xr.open_mfdataset(fsai+'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.0'+str(ifile)+'.cam.h0.FSNSC.203501-206912.nc')
        
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
    SAI_sw_clrR = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                     lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SAI_sw_clrB = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                     lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SAI_sw_clrCA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),
                                      lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 
    
    ##################################################
    ################################################
    
    SAI_CRE_sw.append(SAI_sw-SAI_sw_clr)
    SAI_CRE_swR.append(SAI_swR-SAI_sw_clrR)
    SAI_CRE_swB.append(SAI_swB-SAI_sw_clrB)
    SAI_CRE_swCA.append(SAI_swCA-SAI_sw_clrCA)
    

SAI_CRE_sw  = np.nanmean(np.array(SAI_CRE_sw),axis=0)
SAI_CRE_swR = np.array(SAI_CRE_swR)
SAI_CRE_swB = np.array(SAI_CRE_swB)
SAI_CRE_swCA = np.array(SAI_CRE_swCA)


SAI_CRE = SAI_CRE_sw+SAI_CRE_lw
SAI_CRE_R = SAI_CRE_swR+SAI_CRE_lwR
SAI_CRE_B = SAI_CRE_swB+SAI_CRE_lwB
SAI_CRE_CA = SAI_CRE_swCA+SAI_CRE_lwCA


####################################################################################
#######################################################################################
####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FLNS/'
ddirs  = os.listdir(fssp)

fileN = [1,2,3,4,5,6,7,8,9,10]

SSP245_CRE_lw = []; SSP245_CRE_lwR = [];
SSP245_CRE_lwB = []; SSP245_CRE_lwCA = []

for ifile in fileN:
    print(ifile)
    fssp   = 'D:/SAI_Data/SSP245/FLNS/'
    if ifile <=9:
        SSP245_lw_net = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00'+str(ifile)+'.cam.h0.FLNS.201501-206912.nc')
    else:
        SSP245_lw_net = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0'+str(ifile)+'.cam.h0.FLNS.201501-206912.nc')

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
    SSP245_lwR = SSP245_lw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                   lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SSP245_lwB = SSP245_lw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                   lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SSP245_lwCA = SSP245_lw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                    lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 
        
    ###############################################################################
    ###############################################################################
    fssp   = 'D:/SAI_Data/SSP245/FLNSC/'
    ddirs  = os.listdir(fssp)
    
    if ifile<=9:
        SSP245_lw_net_clr = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00'+str(ifile)+'.cam.h0.FLNSC.201501-206912.nc')
    else:
        SSP245_lw_net_clr = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0'+str(ifile)+'.cam.h0.FLNSC.201501-206912.nc')

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
    SSP245_lw_clrR = SSP245_lw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                           lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SSP245_lw_clrB = SSP245_lw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                           lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SSP245_lw_clrCA = SSP245_lw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                            lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 

    ###############################################################
    ##########################################################################
    SSP245_CRE_lw.append(-SSP245_lw +SSP245_lw_clr)
    SSP245_CRE_lwR.append(-SSP245_lwR +SSP245_lw_clrR)
    SSP245_CRE_lwB.append(-SSP245_lwB +SSP245_lw_clrB)
    SSP245_CRE_lwCA.append(-SSP245_lwCA +SSP245_lw_clrCA)


SSP245_CRE_lw = np.nanmean(np.array(SSP245_CRE_lw),axis=0)
SSP245_CRE_lwR = np.array(SSP245_CRE_lwR)
SSP245_CRE_lwB = np.array(SSP245_CRE_lwB)
SSP245_CRE_lwCA = np.array(SSP245_CRE_lwCA)


#####################################################################
###########################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNS/'
ddirs  = os.listdir(fssp)

SSP245_CRE_sw   = []; SSP245_CRE_swR  =[];
SSP245_CRE_swB  = []; SSP245_CRE_swCA = []

for ifile in fileN:
    print(ifile)
    fssp   = 'D:/SAI_Data/SSP245/FSNS/'
    if ifile <=9:
        SSP245_sw_net = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00'+str(ifile)+'.cam.h0.FSNS.201501-206912.nc')
    else:
        SSP245_sw_net = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0'+str(ifile)+'.cam.h0.FSNS.201501-206912.nc')

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
    SSP245_swR = SSP245_sw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                   lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SSP245_swB = SSP245_sw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                   lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SSP245_swCA = SSP245_sw_net.sel(time = slice('2015-01-01','2069-12-31'),
                                    lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

####################################################################################
#######################################################################################
    fssp   = 'D:/SAI_Data/SSP245/FSNSC/'
    ddirs  = os.listdir(fssp)

    if ifile <=9:
        SSP245_sw_net_clr = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.00'+str(ifile)+'.cam.h0.FSNSC.201501-206912.nc')
    else:
        SSP245_sw_net_clr = xr.open_mfdataset(fssp+'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.0'+str(ifile)+'.cam.h0.FSNSC.201501-206912.nc')

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
    SSP245_sw_clrR = SSP245_sw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                           lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 
    
    #################### black box 
    lat = laC[np.where((laC>=16)&(laC<=22))[0]]
    lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
    SSP245_sw_clrB = SSP245_sw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                           lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 
    
    #################### Green box 
    lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
    lon = loC[np.where((loC>=10)&(loC<=26))[0]]
    SSP245_sw_clrCA = SSP245_sw_net_clr.sel(time = slice('2015-01-01','2069-12-31'),
                                            lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 

    SSP245_CRE_sw.append(SSP245_sw- SSP245_sw_clr)
    SSP245_CRE_swR.append(SSP245_swR- SSP245_sw_clrR)
    SSP245_CRE_swB.append(SSP245_swB- SSP245_sw_clrB)
    SSP245_CRE_swCA.append(SSP245_swCA- SSP245_sw_clrCA)
    
SSP245_CRE_sw = np.nanmean(np.array(SSP245_CRE_sw),axis=0)
SSP245_CRE_swR = np.array(SSP245_CRE_swR)
SSP245_CRE_swB = np.array(SSP245_CRE_swB)
SSP245_CRE_swCA = np.array(SSP245_CRE_swCA)

SSP245_CRE    = SSP245_CRE_sw + SSP245_CRE_lw
SSP245_CRE_R  = SSP245_CRE_swR + SSP245_CRE_lwR
SSP245_CRE_B  = SSP245_CRE_swB + SSP245_CRE_lwB
SSP245_CRE_CA = SSP245_CRE_swCA + SSP245_CRE_lwCA


###############################################################################
###############################################################################

SAII = [SAI_CRE_sw, SAI_CRE_lw, SAI_CRE]
SSP245 = [SSP245_CRE_sw, SSP245_CRE_lw, SSP245_CRE]


PVAL1 = []
for i in np.arange(0 ,3): #ttest_rel
    statres, pval1 = ttest_ind(SAII[i], SSP245[i], axis=0, nan_policy='propagate', alternative='two-sided')
    spy1 = np.where(pval1 < 0.05 , 1,np.nan)
    PVAL1.append(spy1)

PVAL = np.array(PVAL1)

################################################################
################################################################

##plt.style.use('ggplot')

Bias_net = np.nanmean(SAI_CRE, axis=0)-np.nanmean(SSP245_CRE, axis=0)
Bias_net_sw = np.nanmean(SAI_CRE_sw, axis=0) - np.nanmean(SSP245_CRE_sw, axis=0)
Bias_net_lw = np.nanmean(SAI_CRE_lw, axis=0) - np.nanmean(SSP245_CRE_lw, axis=0)

########## percentage of change

Perc_Bias_net    = -100*(Bias_net)/abs(np.nanmean(SSP245_CRE, axis=0))
Perc_Bias_net_sw = -100*(Bias_net_sw)/abs(np.nanmean(SSP245_CRE_sw, axis=0))
Perc_Bias_net_lw = 100*(Bias_net_lw)/abs(np.nanmean(SSP245_CRE_lw, axis=0))
Perc_BIAS = [Perc_Bias_net_sw,Perc_Bias_net_lw,Perc_Bias_net]



################################################################################
########### standard deviation accross simulations
################################################################################
################ SWA
std_ssp245_lwR = np.std(SSP245_CRE_lwR, axis=0)
std_ssp245_swR = np.std(SSP245_CRE_swR, axis=0)
std_ssp245_netR = np.std(SSP245_CRE_R, axis=0)

std_SAI_lwR = np.std(SAI_CRE_lwR, axis=0)
std_SAI_swR = np.std(SAI_CRE_swR, axis=0)
std_SAI_netR = np.std(SAI_CRE_R, axis=0)

########## SAH
std_ssp245_lwB = np.std(SSP245_CRE_lwB, axis=0)
std_ssp245_swB = np.std(SSP245_CRE_swB, axis=0)
std_ssp245_netB = np.std(SSP245_CRE_B, axis=0)

std_SAI_lwB = np.std(SAI_CRE_lwB, axis=0)
std_SAI_swB = np.std(SAI_CRE_swB, axis=0)
std_SAI_netB = np.std(SAI_CRE_B, axis=0)

############ CA
std_ssp245_lwCA = np.std(SSP245_CRE_lwCA, axis=0)
std_ssp245_swCA = np.std(SSP245_CRE_swCA, axis=0)
std_ssp245_netCA = np.std(SSP245_CRE_CA, axis=0)

std_SAI_lwCA = np.std(SAI_CRE_lwCA, axis=0)
std_SAI_swCA = np.std(SAI_CRE_swCA, axis=0)
std_SAI_netCA = np.std(SAI_CRE_CA, axis=0)


################################################################################
########### calculate the interannual standard deviation
################################################################################
################ SWA
Int_std_ssp245_lwR = np.std(np.mean(SSP245_CRE_lwR, axis=0))
Int_std_ssp245_swR = np.std(np.mean(SSP245_CRE_swR, axis=0))
Int_std_ssp245_netR = np.std(np.mean(SSP245_CRE_R, axis=0))

Int_std_SAI_lwR = np.std(np.mean(SAI_CRE_lwR, axis=0))
Int_std_SAI_swR = np.std(np.mean(SAI_CRE_swR, axis=0))
Int_std_SAI_netR = np.std(np.mean(SAI_CRE_R, axis=0))

########## SAH
Int_std_ssp245_lwB = np.std(np.mean(SSP245_CRE_lwB, axis=0))
Int_std_ssp245_swB = np.std(np.mean(SSP245_CRE_swB, axis=0))
Int_std_ssp245_netB = np.std(np.mean(SSP245_CRE_B, axis=0))

Int_std_SAI_lwB = np.std(np.mean(SAI_CRE_lwB, axis=0))
Int_std_SAI_swB = np.std(np.mean(SAI_CRE_swB, axis=0))
Int_std_SAI_netB = np.std(np.mean(SAI_CRE_B, axis=0))

############ CA
Int_std_ssp245_lwCA = np.std(np.mean(SSP245_CRE_lwCA, axis=0))
Int_std_ssp245_swCA = np.std(np.mean(SSP245_CRE_swCA, axis=0))
Int_std_ssp245_netCA = np.std(np.mean(SSP245_CRE_CA, axis=0))

Int_std_SAI_lwCA = np.std(np.mean(SAI_CRE_lwCA, axis=0))
Int_std_SAI_swCA = np.std(np.mean(SAI_CRE_swCA, axis=0))
Int_std_SAI_netCA = np.std(np.mean(SAI_CRE_CA, axis=0))

##########################################################################
###########################################################################



IND = ['a) ΔSW CRE','b) ΔLW CRE','c) ΔNet CRE']

clevv = np.arange(-4,4.2,0.2)
clevvPerc = np.arange(-13,13.5,0.5)
levels = clevvPerc

cmap = plt.get_cmap('bwr')
cmap_r = plt.get_cmap('PuOr')

fig = plt.figure(figsize=(28,16))
gs = fig.add_gridspec(4, 3, bottom=0.0, top=0.85, left=0.05, 
                      right=0.5, wspace=0.22, hspace=0.30)

ax0 = fig.add_subplot(gs[0, 0]) # ,projection = ccrs.PlateCarree()
cs=ax0.contourf(LONMod, LATMod, Bias_net_sw, clevv,
      cmap= cmap ,extend='both') #,transform = ccrs.PlateCarree()

css =ax0.contour(LONMod, LATMod, Perc_Bias_net_sw, levels[::2],
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax0.scatter(LONMod*PVAL[0], LATMod*PVAL[0], s=10, marker='.', color = 'black')
ax0.clabel(css, css.levels[::2], fontsize = 15, inline =True,colors='k',fmt ='%0.0f') # "%0.0f") 
#ax0.coastlines(linewidth=4)
ax0.set_title(IND[0],fontweight = 'bold',fontsize=20)
ax0.set_xticks(np.arange(-20,40,6),)#crs=ccrs.PlateCarree()
lon_formatter = cticker.LongitudeFormatter()
ax0.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax0.set_yticks(np.arange(-10,30,4), )#crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax0.yaxis.set_major_formatter(lat_formatter)
#ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax0.axes.get_xaxis().set_ticklabels
ax0.axes.get_yaxis().set_ticklabels
ax0.axes.axis('tight')
ax0.add_patch(Rectangle((-12, 7), 18, 6, fc ='none', 
                  color = 'red', linewidth =4, linestyle ='-'))

ax0.add_patch(Rectangle((-8, 16), 12, 6, fc ='none', 
                  color = 'black', linewidth =4, linestyle = '-')) 
ax0.add_patch(Rectangle((10, -2), 16, 7, fc ='none', 
                  color = 'green', linewidth =4,linestyle = '-')) 


##############################################################################
###########################################################################
clevvPerc = np.arange(-13,14,1)
levels = clevvPerc

ax1 = fig.add_subplot(gs[0, 1])#, projection=ccrs.PlateCarree()
cs=ax1.contourf(LONMod, LATMod, Bias_net_lw,clevv,cmap= cmap ,extend='both') #transform = ccrs.PlateCarree(),
css =ax1.contour(LONMod, LATMod, Perc_Bias_net_lw, levels[::2],
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax1.clabel(css, levels[::2], fontsize = 15, inline =True, colors='k',fmt = "%0.0f") 
ax1.scatter(LONMod*PVAL[1], LATMod*PVAL[1], s=10, marker='.', color = 'black')
 
    
#ax1.coastlines(linewidth=4)
ax1.set_title(IND[1],fontweight = 'bold',fontsize=20)
ax1.set_xticks(np.arange(-20,40,6))#, crs=ccrs.PlateCarree()
lon_formatter = cticker.LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax1.set_yticks(np.arange(-10,30,4))#, crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
#ax1.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax1.axes.get_xaxis().set_ticklabels
ax1.axes.get_yaxis().set_ticklabels
ax1.axes.axis('tight')
ax1.add_patch(Rectangle((-12, 7), 18, 6, fc ='none', 
                  color = 'red', linewidth =4, linestyle = '-'))
ax1.add_patch(Rectangle((-8, 16), 12, 6, fc ='none', 
                  color = 'black', linewidth =4, linestyle = '-')) 
ax1.add_patch(Rectangle((10, -2), 16, 7, fc ='none', 
                  color = 'green', linewidth =4, linestyle = '-')) 

##############################################################################
###########################################################################
clevvPerc = np.arange(-13,14,1)
levels = clevvPerc

ax2 = fig.add_subplot(gs[0, 2]) #projection=ccrs.PlateCarree()
cs=ax2.contourf(LONMod, LATMod, Bias_net,clevv, cmap= cmap ,extend='both') # transform = ccrs.PlateCarree(),

css =ax2.contour(LONMod, LATMod, Perc_Bias_net, levels[::2],
                  linewidths = 1,colors='gray', extend='both') #,transform=ccrs.PlateCarree()
ax2.clabel(css, levels[::2], fontsize = 15, inline =True, colors='k',fmt = "%0.0f")   
    
#ax2.coastlines(linewidth=4)
ax2.set_title(IND[2],fontweight = 'bold',fontsize=20)
ax2.set_xticks(np.arange(-20,30,6)) #, crs=ccrs.PlateCarree()
lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax2.set_yticks(np.arange(-10,40,4)) # crs=ccrs.PlateCarree()
lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
#ax2.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax2.axes.get_xaxis().set_ticklabels
ax2.axes.get_yaxis().set_ticklabels
ax2.axes.axis('tight')
ax2.add_patch(Rectangle((-12, 7), 18, 6, fc ='none', 
                  color = 'red', linewidth =4, linestyle = '-'))
ax2.add_patch(Rectangle((-8, 16),  12, 6, fc ='none',
                        color = 'black', linewidth =4, linestyle = '-')) 
ax2.add_patch(Rectangle((10, -2), 16, 7, fc ='none', 
                  color = 'green', linewidth =4, linestyle = '-')) 
ax2.scatter(LONMod*PVAL[2], LATMod*PVAL[2], s=10, marker='.', color = 'black')


cbar_ax = fig.add_axes([0.51, 0.69, 0.006, 0.15]) 
cbar=fig.colorbar(cs, cbar_ax, orientation='vertical',extendrect = 'True')
cbar.set_label('W/m²', fontsize=20,fontweight = 'bold')

######################################
########################################
ax3 = fig.add_subplot(gs[1, 0])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_swR, axis=0), 10, 2)
ax3.plot(X, fitt, 'r--', linewidth=3.5)
ax3.fill_between(X, fitt-Int_std_ssp245_swR, fitt+Int_std_ssp245_swR,color ='r',alpha=0.3)
for iss in np.arange(0,10):
    ax3.plot(X, SSP245_CRE_swR[iss], 'r--', linewidth=0.3)

##ax3.plot(X, fitt, 'r-', linewidth=1)  
X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_swR,axis=0), 10, 2)
ax3.plot(X, fitt ,'k-', linewidth=3.5) #X,SAI_CRE_swB,'k-',
for iss in np.arange(0,10):
    ax3.plot(X, SAI_CRE_swR[iss], 'k--', linewidth=0.3) 
ax3.fill_between(X, fitt-Int_std_SAI_swR, fitt+Int_std_SAI_swR,color ='k',alpha=0.3)


ax3.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
ax3.set_ylabel('CRE (W/m²)', size=20, fontweight ='bold')

#fig.text(0.15,0.63,'SWA',fontsize=20, fontweight ='bold')
#ax3.set_xlabel('Years', size=20, fontweight ='bold')
ax3.set_title('d) SW', size=20, fontweight ='bold')

#############################################################
#############################################################
ax4 = fig.add_subplot(gs[1, 1])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_lwR,axis=0), 10, 2)
ax4.plot(X, fitt ,'r--', linewidth=3.5) 
for iss in np.arange(0,10):
    ax4.plot(X, SSP245_CRE_lwR[iss], 'r--', linewidth=0.3)
    
ax4.fill_between(X, fitt-Int_std_ssp245_lwR, fitt+Int_std_ssp245_lwR,color ='r',alpha=0.3)

X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_lwR,axis=0), 10, 2)
ax4.plot(X, fitt ,'k-', linewidth=3.5) #X,SAI_CRE_lwB,'k-',
for iss in np.arange(0,10):
    ax4.plot(X, SAI_CRE_lwR[iss], 'k--', linewidth=0.3)
ax4.fill_between(X, fitt-Int_std_SAI_lwR, fitt+Int_std_SAI_lwR,color ='k',alpha=0.3)
ax4.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
#ax4.set_xlabel('Years', size=20, fontweight ='bold')
ax4.set_title('e) LW', size=20, fontweight ='bold')
# ax4.plot([2035, 2035], [-10, -8.5] ,'k-', linewidth=3.5) 
# #ax0.plot([2035], [-56], 'o')
# ax4.annotate('2035', xy=(2035, -10), xytext=(2050, -9.5),
#             arrowprops=dict(facecolor='black', shrink=0.05), fontweight ='bold')

###########################################################################
######################################################################
ax5 = fig.add_subplot(gs[1, 2])
X1 = np.arange(2015, 2070)
fitt1 = savgol_filter(np.mean(SSP245_CRE_R,axis=0), 10, 2)
ax5.plot(X1, fitt1 ,'r--', linewidth=3.5) 
X2 = np.arange(2035, 2070)

fitt2 = savgol_filter(np.mean(SAI_CRE_R,axis=0), 10, 2)
ax5.plot(X2, fitt2 ,'k-', linewidth=3.5) #X,SAI_CRE_B,'k-',
for iss in np.arange(0,10):
    ax5.plot(X1, SSP245_CRE_R[iss], 'r--', linewidth=0.3)
ax5.fill_between(X1, fitt1-Int_std_ssp245_netR, fitt1+Int_std_ssp245_netR,color ='r',alpha=0.3)
for iss in np.arange(0,10):
    ax5.plot(X2, SAI_CRE_R[iss], 'k--', linewidth=0.3)
ax5.fill_between(X2, fitt2-Int_std_SAI_netR, fitt2+Int_std_SAI_netR,color ='k',alpha=0.3)

ax5.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
#ax5.set_xlabel('Years', size=20, fontweight ='bold')
ax5.set_title('f) Net', size=20, fontweight ='bold')
ax5.legend(['SSP2-4.5','ARISE-SAI-1.5'], loc= 'best',
          shadow = False, ncol = 1, prop={'size':15}) #.get_frame().set_facecolor('b')

###################################################################
##################### black box

ax6 = fig.add_subplot(gs[2, 0])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_swB,axis=0), 10, 2)
ax6.plot(X, fitt ,'r--', linewidth=3.5) 
for iss in np.arange(0,10):
     ax6.plot(X, SSP245_CRE_swB[iss], 'r--', linewidth=0.3)
ax6.fill_between(X, fitt-Int_std_ssp245_swB, fitt+Int_std_ssp245_swB,color ='r',alpha=0.3)

X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_swB,axis=0), 10, 2)
for iss in np.arange(0,10):
    ax6.plot(X, SAI_CRE_swB[iss], 'k--', linewidth=0.3)   
ax6.plot(X, fitt ,'k-', linewidth=3.5) #X,,'k-',
ax6.fill_between(X, fitt-Int_std_SAI_swB, fitt+Int_std_SAI_swB,color ='k',alpha=0.3)
ax6.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
ax6.set_ylabel('CRE (W/m²)', size=20, fontweight ='bold')
#ax6.set_xlabel('Years', size=20, fontweight ='bold')
ax6.set_title('g) SW', size=20, fontweight ='bold')

#############################################################
#############################################################
ax7 = fig.add_subplot(gs[2, 1])
X1 = np.arange(2015, 2070)

fitt1 = savgol_filter(np.mean(SSP245_CRE_lwB,axis=0), 10, 2)
ax7.plot(X1, fitt1 ,'r--', linewidth=3.5) 
X2 = np.arange(2035, 2070)
fitt2 = savgol_filter(np.mean(SAI_CRE_lwB,axis=0), 10, 2)
ax7.plot(X2, fitt2 ,'k-', linewidth=3.5) #X,,'k-',

for iss in np.arange(0,10):
    ax7.plot(X1, SSP245_CRE_lwB[iss], 'r--', linewidth=0.3)
ax7.fill_between(X1, fitt1-Int_std_ssp245_lwB, fitt1+Int_std_ssp245_lwB,color ='r',alpha=0.3)
for iss in np.arange(0,10):
    ax7.plot(X2, SAI_CRE_lwB[iss], 'k--', linewidth=0.3)
ax7.fill_between(X2, fitt2-Int_std_SAI_lwB, fitt2+Int_std_SAI_lwB,color ='k',alpha=0.3)

ax7.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
#ax7.set_xlabel('Years', size=20, fontweight ='bold')
ax7.set_title('h) LW', size=20, fontweight ='bold')
ax7.legend(['SSP2-4.5','ARISE-SAI-1.5'], loc= 'best',
          shadow = False, ncol = 1, prop={'size':15}) #.get_frame().set_facecolor('b')
# ax7.plot([2035, 2035], [-10, -8.5] ,'k-', linewidth=3.5) 
# #ax0.plot([2035], [-56], 'o')
# ax7.annotate('2035', xy=(2035, -10), xytext=(2050, -9.5),
#             arrowprops=dict(facecolor='black', shrink=0.05), fontweight ='bold')

###########################################################################
######################################################################

ax8 = fig.add_subplot(gs[2, 2])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_B, axis=0),20, 4)
ax8.plot(X, fitt ,'r--', linewidth=3.5) 
for iss in np.arange(0,10):
    ax8.plot(X, SSP245_CRE_B[iss], 'r--', linewidth=0.3)

ax8.fill_between(X, fitt-Int_std_ssp245_netB, fitt+Int_std_ssp245_netB,color ='r',alpha=0.3)
X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_B,axis=0), 10, 2)
ax8.plot(X, fitt ,'k-', linewidth=3.5) #X,,'k-',
for iss in np.arange(0,10):
    ax8.plot(X, SAI_CRE_B[iss], 'k--', linewidth=0.3)
ax8.fill_between(X, fitt-Int_std_SAI_netB, fitt+Int_std_SAI_netB,color ='k',alpha=0.3)
ax8.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
#ax8.set_xlabel('Years', size=20, fontweight ='bold')
ax8.set_title('i) Net', size=20, fontweight ='bold')

# ax8.plot([2035, 2035], [-66, -58] ,'k-', linewidth=3.5) #X
# ax8.annotate('2035', xy=(2035, -65), xytext=(2050, -63),
#             arrowprops=dict(facecolor='black', shrink=0.05), fontweight ='bold')
###################################################################
################################################################


ax9 = fig.add_subplot(gs[3, 0])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_swCA,axis=0), 10, 2)
ax9.plot(X, fitt ,'r--', linewidth=3.5) 
for iss in np.arange(0,10):
    ax9.plot(X, SSP245_CRE_swCA[iss], 'r--', linewidth=.3)

ax9.fill_between(X, fitt-Int_std_ssp245_swCA, fitt+Int_std_ssp245_swCA,color ='r',alpha=0.3)

X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_swCA,axis=0), 10, 2)
ax9.plot(X, fitt ,'k-', linewidth=3.5) #X,,'k-',
for iss in np.arange(0,10):
    ax9.plot(X, SAI_CRE_swCA[iss], 'k--', linewidth=.3)
ax9.fill_between(X, fitt-Int_std_SAI_swCA, fitt+Int_std_SAI_swCA,color ='k',alpha=0.3)

ax9.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
ax9.set_xlabel('Years', size=20, fontweight ='bold')
ax9.set_title('j) SW', size=20, fontweight ='bold')
ax9.set_ylabel('CRE (W/m²)', size=20, fontweight ='bold')
###################################################################
################################################################

ax10 = fig.add_subplot(gs[3, 1])
X1 = np.arange(2015, 2070)
fitt1 = savgol_filter(np.mean(SSP245_CRE_lwCA, axis=0), 10, 2)   
ax10.plot(X1, fitt1 ,'r--', linewidth=3.5) 

X2 = np.arange(2035, 2070)
fitt2 = savgol_filter(np.mean(SAI_CRE_lwCA,axis=0), 10, 2)
ax10.plot(X2, fitt2 ,'k-', linewidth=3.5) #X,,'k-',
for iss in np.arange(0,10):
    ax10.plot(X1, SSP245_CRE_lwCA[iss], 'r--', linewidth=.3)
for iss in np.arange(0,10):
    ax10.plot(X2, SAI_CRE_lwCA[iss], 'k--', linewidth=0.3)
ax10.fill_between(X1, fitt1-Int_std_ssp245_lwCA, fitt1+Int_std_ssp245_lwCA,color ='r',alpha=0.3)
ax10.fill_between(X2, fitt2-Int_std_SAI_lwCA, fitt2+Int_std_SAI_lwCA,color ='k',alpha=0.3)

ax10.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
ax10.set_xlabel('Years', size=20, fontweight ='bold')
ax10.set_title('k) LW', size=20, fontweight ='bold')
ax10.legend(['SSP2-4.5','ARISE-SAI-1.5'], loc= 'best',
          shadow = False, ncol = 1, prop={'size':15})

###################################################################
################################################################

ax11 = fig.add_subplot(gs[3, 2])
X = np.arange(2015, 2070)
fitt = savgol_filter(np.mean(SSP245_CRE_CA,axis=0), 10, 2)
ax11.plot(X, fitt ,'r--', linewidth=3.5) 
for iss in np.arange(0,10):
    ax11.plot(X, SSP245_CRE_CA[iss], 'r--', linewidth=0.3)
ax11.fill_between(X, fitt-Int_std_ssp245_netCA, fitt+Int_std_ssp245_netCA,color ='r',alpha=0.3)
X = np.arange(2035, 2070)
fitt = savgol_filter(np.mean(SAI_CRE_CA,axis=0), 10, 2)
ax11.plot(X, fitt ,'k-', linewidth=3.5) #X,,'k-',
for iss in np.arange(0,10):
    ax11.plot(X, SAI_CRE_CA[iss], 'k--', linewidth=0.3)
ax11.fill_between(X, fitt-Int_std_SAI_netCA, fitt+Int_std_SAI_netCA,color ='k',alpha=0.3)
ax11.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)
ax11.set_xlabel('Years', size=20, fontweight ='bold')
ax11.set_title('l) Net', size=20, fontweight ='bold')

fig.text(0.51,0.55,'SWA',fontsize=20, fontweight ='bold',rotation='vertical')
fig.text(0.51,0.27,'SAH',fontsize=20, fontweight ='bold',rotation='vertical')
fig.text(0.51,0.05,'CA',fontsize=20, fontweight ='bold',rotation='vertical')

plt.savefig("D:/SAI_Data/Figure_1.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_1.pdf", dpi=700, bbox_inches='tight')


