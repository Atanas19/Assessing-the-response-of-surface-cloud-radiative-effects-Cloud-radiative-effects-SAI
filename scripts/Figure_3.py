# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:57:47 2024

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

###############################################################################
###############################################################################
########################### SSP2-4.5 #############################################
###############################################################################
###############################################################################
shapefile_path = 'D:/SAI_Data/Africa_Boundaries-shp/Africa_Boundaries.shp'
f = 'D:/SAI_Data/SSP245/LWP/'
ddirsM = os.listdir(f)



fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNS/'
ddirsR = os.listdir(fsai)

SAI_lw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FLNS.203501-206912.nc')
SAI_lw_net.coords['lon'] = (SAI_lw_net.coords['lon'] + 180) % 360 - 180
SAI_lw_net = SAI_lw_net.sortby(SAI_lw_net.lon)

SAI_lw_net = SAI_lw_net.convert_calendar(calendar = 'gregorian', align_on = 'date')

########################################################

laC = SAI_lw_net.lat.values
loC = SAI_lw_net.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
LONMod, LATMod = np.meshgrid(lon,lat)

LONC, LATC = np.meshgrid(lon,lat)
ds  = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FLNS'] # '2035-02-01'

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
##########################################################################
##############################################################################

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_WA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'), lat=lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_SA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_CA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values



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


laC = SAI_lw_net_clr.lat.values
loC = SAI_lw_net_clr.lon.values
lat = laC[np.where((laC>=-10)&(laC <=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]
ds  = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FLNSC']

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

###############################################################################
##########################################################################
##############################################################################

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_clr_WA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_clr_SA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_clr_CA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

SAI_CRE_lw    = -SAI_lw+SAI_lw_clr
SAI_CRE_lw_CA = -SAI_lw_CA+SAI_lw_clr_CA
SAI_CRE_lw_SA = -SAI_lw_SA+SAI_lw_clr_SA
SAI_CRE_lw_WA = -SAI_lw_WA+SAI_lw_clr_WA
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
ds  = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'), lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FSNS']

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


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_WA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_SA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_CA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values


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
ds  = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FSNSC']

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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_clr_WA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_clr_SA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_clr_CA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values


SAI_CRE_sw    = SAI_sw-SAI_sw_clr
SAI_CRE_sw_CA = SAI_sw_CA-SAI_sw_clr_CA
SAI_CRE_sw_SA = SAI_sw_SA-SAI_sw_clr_SA
SAI_CRE_sw_WA = SAI_sw_WA-SAI_sw_clr_WA

SAI_CRE = SAI_CRE_sw+SAI_CRE_lw
SAI_CRE_CA = SAI_CRE_sw_CA+SAI_CRE_lw_CA
SAI_CRE_SA = SAI_CRE_sw_SA+SAI_CRE_lw_SA
SAI_CRE_WA = SAI_CRE_sw_WA+SAI_CRE_lw_WA

####################################################################################
#######################################################################################
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
ds  = SSP245_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FLNS']

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


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_WA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_SA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_CA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNS'].groupby('time.year').mean(dim='time').values


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

ds  = SSP245_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FLNSC']

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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_clr_WA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_clr_SA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_clr_CA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FLNSC'].groupby('time.year').mean(dim='time').values

SSP245_CRE_lw    = -SSP245_lw +SSP245_lw_clr
SSP245_CRE_lw_CA = -SSP245_lw_CA +SSP245_lw_clr_CA
SSP245_CRE_lw_SA = -SSP245_lw_SA +SSP245_lw_clr_SA
SSP245_CRE_lw_WA = -SSP245_lw_WA +SSP245_lw_clr_WA
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
ds  = SSP245_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FSNS']

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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_WA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_SA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_CA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNS'].groupby('time.year').mean(dim='time').values


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
ds  = SSP245_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).groupby('time.year').mean(dim='time')['FSNSC']

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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_clr_WA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_clr_SA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_clr_CA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon)['FSNSC'].groupby('time.year').mean(dim='time').values

SSP245_CRE_sw    = SSP245_sw- SSP245_sw_clr
SSP245_CRE_sw_CA = SSP245_sw_CA- SSP245_sw_clr_CA
SSP245_CRE_sw_SA = SSP245_sw_SA- SSP245_sw_clr_SA
SSP245_CRE_sw_WA = SSP245_sw_WA- SSP245_sw_clr_WA

SSP245_CRE    = SSP245_CRE_sw + SSP245_CRE_lw
SSP245_CRE_CA = SSP245_CRE_sw_CA + SSP245_CRE_lw_CA
SSP245_CRE_SA = SSP245_CRE_sw_SA + SSP245_CRE_lw_SA
SSP245_CRE_WA = SSP245_CRE_sw_WA + SSP245_CRE_lw_WA

###################################################################
#######################################################################

Bias_net = SAI_CRE-SSP245_CRE
Bias_net_sw = SAI_CRE_sw - SSP245_CRE_sw
Bias_net_lw = SAI_CRE_lw - SSP245_CRE_lw

BIAS = [Bias_net_sw,Bias_net_lw,Bias_net]


Bias_net_CA = SAI_CRE_CA-SSP245_CRE_CA
Bias_net_WA = SAI_CRE_WA-SSP245_CRE_WA
Bias_net_SA = SAI_CRE_SA-SSP245_CRE_SA

Bias_net_sw_CA = SAI_CRE_sw_CA - SSP245_CRE_sw_CA
Bias_net_sw_WA = SAI_CRE_sw_WA - SSP245_CRE_sw_WA
Bias_net_sw_SA = SAI_CRE_sw_SA - SSP245_CRE_sw_SA

Bias_net_lw_CA = SAI_CRE_lw_CA - SSP245_CRE_lw_CA
Bias_net_lw_WA = SAI_CRE_lw_WA - SSP245_CRE_lw_WA
Bias_net_lw_SA = SAI_CRE_lw_SA - SSP245_CRE_lw_SA

###############################################################################
###############################################################################
#################### signal to noise ratio
###############################################################################

def fit_values(data):
    fit_data = savgol_filter(data, 10, 2)
    return fit_data

def SNR(predicted, projection):
    Bias = predicted-projection
    STDEV = np.nanstd(projection, axis=0)
    STDEV = np.resize(STDEV,Bias.shape)
    return Bias/STDEV


SNR_net_CA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_CA, SSP245_CRE_CA), axis=2),axis=1))
SNR_net_WA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_WA, SSP245_CRE_WA), axis=2),axis=1))
SNR_net_SA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_SA, SSP245_CRE_SA), axis=2),axis=1))

SNR_sw_CA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_sw_CA, SSP245_CRE_sw_CA), axis=2),axis=1))
SNR_sw_WA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_sw_WA, SSP245_CRE_sw_WA), axis=2),axis=1))
SNR_sw_SA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_sw_SA, SSP245_CRE_sw_SA), axis=2),axis=1))

SNR_lw_CA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_lw_CA, SSP245_CRE_lw_CA), axis=2),axis=1))
SNR_lw_WA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_lw_WA, SSP245_CRE_lw_WA), axis=2),axis=1))
SNR_lw_SA = fit_values(np.nanmean(np.nanmean(SNR(SAI_CRE_lw_SA, SSP245_CRE_lw_SA), axis=2),axis=1))

SNR_net_SP  = np.nanmean(SNR(SAI_CRE, SSP245_CRE), axis=0)
SNR_sw_SP   = np.nanmean(SNR(SAI_CRE_sw, SSP245_CRE_sw), axis=0)
SNR_lw_SP   = np.nanmean(SNR(SAI_CRE_lw, SSP245_CRE_lw), axis=0)

########################################################################
###########################################################################


IND = ['a) SW CRE','b) LW CRE','c) Net CRE']

clevv = np.arange(-3,3.2,0.2)
cmap = plt.get_cmap('bwr')
levels = clevv

fig = plt.figure(figsize=(28,8))
gs = fig.add_gridspec(2, 3, bottom=0.0, top=0.85, left=0.05, 
                      right=0.5, wspace=0.22, hspace=0.30)

ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
cs=ax0.contourf(LONMod, LATMod, SNR_sw_SP,levels,
    transform = ccrs.PlateCarree(), cmap= cmap ,extend='both') 

ax0.scatter(LONMod*np.where(SNR_sw_SP<=-1, 1, np.where(SNR_sw_SP>=1, 1, np.nan)), 
                            LATMod*np.where(SNR_sw_SP<=-1, 1, np.where(SNR_sw_SP>=1, 1, np.nan)), 
                            s=10, marker='.', color = 'black')
  
ax0.coastlines(linewidth=4)
ax0.set_title(IND[0],fontweight = 'bold',fontsize=20)
ax0.set_xticks(np.arange(-20,40,6), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax0.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax0.set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax0.yaxis.set_major_formatter(lat_formatter)
ax0.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax0.axes.get_xaxis().set_ticklabels
ax0.axes.get_yaxis().set_ticklabels
ax0.axes.axis('tight')
ax0.add_patch(Rectangle((-12, 7), 18, 6, fc ='none', 
                  color = 'red', linewidth =4, linestyle = '-'))

ax0.add_patch(Rectangle((-8, 16), 12, 6, fc ='none', 
                  color = 'black', linewidth =4, linestyle = '-')) 
ax0.add_patch(Rectangle((10, -2), 16, 7, fc ='none', 
                  color = 'green', linewidth =4, linestyle = '-')) 
#ax0.scatter(LONMod*PVAL[1], LATMod*PVAL[0], s=10, marker='.', color = 'black')
##############################################################################
###########################################################################

ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
cs=ax1.contourf(LONMod, LATMod, SNR_lw_SP,clevv,
    transform = ccrs.PlateCarree(), cmap= cmap ,extend='both')   

ax1.scatter(LONMod*np.where(SNR_lw_SP<=-1, 1, np.where(SNR_lw_SP>=1, 1, np.nan)), 
                            LATMod*np.where(SNR_lw_SP<=-1, 1, np.where(SNR_lw_SP>=1, 1, np.nan)), 
                            s=10, marker='.', color = 'black')
    
ax1.coastlines(linewidth=4)
ax1.set_title(IND[1],fontweight = 'bold',fontsize=20)
ax1.set_xticks(np.arange(-20,40,6), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax1.set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
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

ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
cs=ax2.contourf(LONMod, LATMod, SNR_net_SP,clevv,
    transform = ccrs.PlateCarree(), cmap= cmap ,extend='both')   
ax2.scatter(LONMod*np.where(SNR_net_SP<=-1, 1, np.where(SNR_net_SP>=1, 1, np.nan)), 
                            LATMod*np.where(SNR_net_SP<=-1, 1, np.where(SNR_net_SP>=1, 1, np.nan)), 
                            s=10, marker='.', color = 'black')  
ax2.coastlines(linewidth=4)
ax2.set_title(IND[2],fontweight = 'bold',fontsize=20)
ax2.set_xticks(np.arange(-20,30,6), crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax2.xaxis.set_major_formatter(lon_formatter)
# Define the yticks for latitude
ax2.set_yticks(np.arange(-10,40,4), crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
ax2.axes.get_xaxis().set_ticklabels
ax2.axes.get_yaxis().set_ticklabels
ax2.axes.axis('tight')
ax2.add_patch(Rectangle((-12, 7),  18, 6, fc ='none', 
                  color = 'red', linewidth =4, linestyle = '-'))

ax2.add_patch(Rectangle((-8, 16), 12, 6, fc ='none', 
                  color = 'black', linewidth =4, linestyle = '-')) 
ax2.add_patch(Rectangle((10, -2), 16, 7, fc ='none', 
                  color = 'green', linewidth =4, linestyle = '-')) 

cbar_ax = fig.add_axes([0.51, 0.47, 0.006, 0.4]) 
cbar=fig.colorbar(cs, cbar_ax, orientation='vertical',extendrect = 'True')
cbar.set_label('SNR', fontsize=20,fontweight = 'bold')

#################################################################
##################################################################

year = np.arange(2035, 2070)
ax3 = fig.add_subplot(gs[1, 0])

ax3.plot(year, SNR_net_WA ,'r-',linewidth=1.5) 
ax3.plot(year, SNR_sw_WA  ,'k-', linewidth=1.5)
ax3.plot(year, SNR_lw_WA ,'m-', linewidth=1.5)
ax3.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(year, SNR_net_SA ,'r-',linewidth=1.5) 
ax4.plot(year, SNR_sw_SA  ,'k-', linewidth=1.5)
ax4.plot(year, SNR_lw_SA ,'m-', linewidth=1.5)
ax4.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)

ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(year, SNR_net_CA ,'r-',linewidth=1.5) 
ax5.plot(year, SNR_sw_CA  ,'k-', linewidth=1.5)
ax5.plot(year, SNR_lw_CA ,'m-', linewidth=1.5)
ax5.grid(True, 'major', 'both', ls='--', lw=1, c='k', alpha=.3)


ax3.set_ylabel('SNR', fontsize=20, fontweight = 'bold')
ax4.set_ylabel('SNR', fontsize=20, fontweight = 'bold')
ax5.set_ylabel('SNR', fontsize=20, fontweight = 'bold')

ax3.set_xlabel('Year', fontsize=20, fontweight = 'bold')
ax4.set_xlabel('Year', fontsize=20, fontweight = 'bold')
ax5.set_xlabel('Year', fontsize=20, fontweight = 'bold')

ax3.set_title('d) SWA', fontsize=20, fontweight = 'bold')
ax4.set_title('e) SAH', fontsize=20, fontweight = 'bold')
ax5.set_title('f) CA', fontsize=20, fontweight = 'bold')

ax5.legend(('NET', 'SW','LW'),loc='best',
          fancybox=True,  shadow=False, ncol=1,prop={'size':15})

plt.savefig("D:/SAI_Data/Figure_3.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_3.pdf", dpi=700, bbox_inches='tight')

