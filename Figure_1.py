# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:01:24 2024

@author: ad7gb
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import scipy.interpolate as inter 
from matplotlib.patches import Rectangle

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

TT = pd.date_range('2035-02-01', '2071-01-31', freq='ME')
SAI_lw_net = SAI_lw_net.assign_coords({'time':SAI_lw_net['time']-SAI_lw_net['time']+ TT})
SAI_lw_net = SAI_lw_net.drop('time_bnds')

#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SAI_lw_net = SAI_lw_net.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SAI_lw_net = SAI_lw_net.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SAI_lw_net.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SAI_lw_net_clip = SAI_lw_net.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC = SAI_lw_net_clip.lat.values
loC = SAI_lw_net_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONC, LATC = np.meshgrid(lon,lat)
dsB = SAI_lw_net_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FLNS']
ds  = SAI_lw_net_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FLNS']

SAI_lw = ds.values
###############################################################################
###############################################################################
fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNSC/'
ddirsR = os.listdir(fsai)

SAI_lw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FLNSC.203501-206912.nc')
SAI_lw_net_clr.coords['lon'] = (SAI_lw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_lw_net_clr = SAI_lw_net_clr.sortby(SAI_lw_net_clr.lon)

TT = pd.date_range('2035-02-01', '2071-01-31', freq='ME')
SAI_lw_net_clr = SAI_lw_net_clr.assign_coords({'time':SAI_lw_net_clr['time']-SAI_lw_net_clr['time']+ TT})
SAI_lw_net_clr = SAI_lw_net_clr.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SAI_lw_net_clr = SAI_lw_net_clr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SAI_lw_net_clr = SAI_lw_net_clr.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SAI_lw_net_clr.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SAI_lw_net_clr_clip = SAI_lw_net_clr.rio.clip(geometries, mskk.crs)
###########################################################
####################################################


laC = SAI_lw_net_clr_clip.lat.values
loC = SAI_lw_net_clr_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC <=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

dsB = SAI_lw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FLNSC']
ds  = SAI_lw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FLNSC']

SAI_lw_clr = ds.values

SAI_CRE_lw = -SAI_lw+SAI_lw_clr

#####################################################################
###########################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNS/'
ddirsR = os.listdir(fsai)

SAI_sw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNS.203501-206912.nc')

SAI_sw_net.coords['lon'] = (SAI_sw_net.coords['lon'] + 180) % 360 - 180
SAI_sw_net = SAI_sw_net.sortby(SAI_sw_net.lon)
SAI_sw_net = SAI_sw_net.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SAI_sw_net = SAI_sw_net.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SAI_sw_net = SAI_sw_net.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SAI_sw_net.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SAI_sw_net_clip = SAI_sw_net.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC  = SAI_sw_net_clip.lat.values
loC  = SAI_sw_net_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)

dsB = SAI_sw_net_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FSNS']
ds  = SAI_sw_net_clip.sel(time = slice('2035-01-01','2070-12-31'), lat = lat, lon = lon).mean(dim='time')['FSNS']

SAI_sw = ds.values
####################################################################################
#######################################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNSC/'
ddirsR = os.listdir(fsai)

SAI_sw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNSC.203501-206912.nc')

SAI_sw_net_clr.coords['lon'] = (SAI_sw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_sw_net_clr = SAI_sw_net_clr.sortby(SAI_sw_net_clr.lon)
SAI_sw_net_clr = SAI_sw_net_clr.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SAI_sw_net_clr = SAI_sw_net_clr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SAI_sw_net_clr = SAI_sw_net_clr.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SAI_sw_net_clr.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SAI_sw_net_clr_clip = SAI_sw_net_clr.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC  = SAI_sw_net_clr_clip.lat.values
loC  = SAI_sw_net_clr_clip.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)

dsB = SAI_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FSNSC']
ds  = SAI_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FSNSC']

SAI_sw_clr = ds.values

SAI_CRE_sw = SAI_sw-SAI_sw_clr
SAI_CRE = SAI_CRE_sw+SAI_CRE_lw


####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FLNS/'
ddirs  = os.listdir(fssp)

SSP245_lw_net = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-005.cam.h0.FLNS.201501-210012.nc')
SSP245_lw_net.coords['lon'] = (SSP245_lw_net.coords['lon'] + 180) % 360 - 180
SSP245_lw_net = SSP245_lw_net.sortby(SSP245_lw_net.lon)

TT = pd.date_range('2015-01-01', '2100-12-31', freq='ME')
SSP245_lw_net = SSP245_lw_net.assign_coords({'time':SSP245_lw_net['time']-SSP245_lw_net['time']+ TT})
SSP245_lw_net = SSP245_lw_net.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SSP245_lw_net = SSP245_lw_net.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SSP245_lw_net = SSP245_lw_net.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SSP245_lw_net.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SSP245_lw_net_clip = SSP245_lw_net.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC = SSP245_lw_net_clip.lat.values
loC = SSP245_lw_net_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONC, LATC = np.meshgrid(lon,lat)

dsB = SSP245_lw_net_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FLNS']
ds  = SSP245_lw_net_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FLNS']

SSP245_lw = ds.values


###############################################################################
###############################################################################
fssp   = 'D:/SAI_Data/SSP245/FLNSC/'
ddirs  = os.listdir(fssp)

SSP245_lw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-005.cam.h0.FLNSC.201501-210012.nc')
SSP245_lw_net_clr.coords['lon'] = (SSP245_lw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_lw_net_clr = SSP245_lw_net_clr.sortby(SSP245_lw_net_clr.lon)

TT = pd.date_range('2015-01-01', '2100-12-31', freq='ME')
SSP245_lw_net_clr = SSP245_lw_net_clr.assign_coords({'time':SSP245_lw_net_clr['time']-SSP245_lw_net_clr['time']+ TT})
SSP245_lw_net_clr = SSP245_lw_net_clr.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SSP245_lw_net_clr = SSP245_lw_net_clr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SSP245_lw_net_clr = SSP245_lw_net_clr.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SSP245_lw_net_clr.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SSP245_lw_net_clr_clip = SSP245_lw_net_clr.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC = SSP245_lw_net_clr_clip.lat.values
loC = SSP245_lw_net_clr_clip.lon.values
lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

dsB = SSP245_lw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FLNSC']
ds  = SSP245_lw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FLNSC']

SSP245_lw_clr = ds.values

SSP245_CRE_lw = -SSP245_lw +SSP245_lw_clr
#####################################################################
###########################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNS/'
ddirs  = os.listdir(fssp)

SSP245_sw_net = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-005.cam.h0.FSNS.201501-210012.nc')

SSP245_sw_net.coords['lon'] = (SSP245_sw_net.coords['lon'] + 180) % 360 - 180
SSP245_sw_net = SSP245_sw_net.sortby(SSP245_sw_net.lon)
SSP245_sw_net = SSP245_sw_net.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SSP245_sw_net = SSP245_sw_net.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SSP245_sw_net = SSP245_sw_net.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SSP245_sw_net.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SSP245_sw_net_clr_clip = SSP245_sw_net.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC  = SSP245_sw_net_clr_clip.lat.values
loC  = SSP245_sw_net_clr_clip.lon.values

lat = laC[np.where((laC>=-10)& (laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)

dsB = SSP245_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FSNS']
ds  = SSP245_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FSNS']

SSP245_sw = ds.values
####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNSC/'
ddirs  = os.listdir(fssp)

SSP245_sw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-005.cam.h0.FSNSC.201501-210012.nc')

SSP245_sw_net_clr.coords['lon'] = (SSP245_sw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_sw_net_clr = SSP245_sw_net_clr.sortby(SSP245_sw_net_clr.lon)
SSP245_sw_net_clr = SSP245_sw_net_clr.drop('time_bnds')
#################### mask the data 
msk = gpd.read_file(shapefile_path, crs = "epsg:4326")
# Set spatial dimensions and CRS
SSP245_sw_net_clr = SSP245_sw_net_clr.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
SSP245_sw_net_clr = SSP245_sw_net_clr.rio.write_crs("epsg:4326", inplace=True)

# Load the shapefile using geopandas
mskk = msk.to_crs(SSP245_sw_net_clr.rio.crs)
# Extract geometries from the GeoDataFrame
geometries = mskk.geometry
# Clip the dataset using the geometries
SSP245_sw_net_clr_clip = SSP245_sw_net_clr.rio.clip(geometries, mskk.crs)
###########################################################
####################################################

laC  = SSP245_sw_net_clr_clip.lat.values
loC  = SSP245_sw_net_clr_clip.lon.values

lat = laC[np.where((laC>=-10)&(laC<=30))[0]]
lon = loC[np.where((loC>=-20)&(loC<=40))[0]]

LONMod, LATMod = np.meshgrid(lon,lat)

dsB = SSP245_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon)['FSNSC']
ds  = SSP245_sw_net_clr_clip.sel(time = slice('2035-01-01','2070-12-31'),lat = lat, lon = lon).mean(dim='time')['FSNSC']

SSP245_sw_clr = ds.values

SSP245_CRE_sw = SSP245_sw - SSP245_sw_clr
SSP245_CRE = SSP245_CRE_sw + SSP245_CRE_lw


###############################################################################
###############################################################################

Bias_net    = SAI_CRE - SSP245_CRE
Bias_net_sw = SAI_CRE_sw - SSP245_CRE_sw
Bias_net_lw = SAI_CRE_lw - SSP245_CRE_lw

SAI = [SAI_CRE_sw, SAI_CRE_lw,SAI_CRE]
SSP245 = [SSP245_CRE_sw, SSP245_CRE_lw, SSP245_CRE]
BIAS = [Bias_net_sw,Bias_net_lw,Bias_net]

###############################################################################
###############################################################################

Perc_Bias_net    = np.array([Bias_net/abs(SSP245_CRE) if SSP245_CRE.any() !=0 else 0])
Perc_Bias_net_sw = np.array([Bias_net_sw/abs(SSP245_CRE_sw) if SSP245_CRE_sw.any() !=0 else 0])
Perc_Bias_net_lw = np.array([Bias_net_lw/abs(SSP245_CRE_lw) if SSP245_CRE_lw.any() !=0 else 0])

Perc_BIAS = [Perc_Bias_net_sw,Perc_Bias_net_lw,Perc_Bias_net]

###############################################################################
###############################################################################
# PVAL1=[]

# for i in np.arange(0 ,3): #ttest_rel
#     statres, pval1 = ttest_ind(DATAA[i], MODELL[i], axis=0, nan_policy='propagate', alternative='two-sided')
#     spy1 = np.where(pval1 < 0.05, np.nan, 1)
#     PVAL1.append(spy1)


################################################################
################################################################

LONC, LATC = np.meshgrid(lon,lat)

clev1 = np.arange(-120,130,10)
clevk = np.arange(-4, 4.2, 0.2)
cmap = plt.get_cmap('seismic') #coolwarm_r; viridis_r
# cmap1 = plt.get_cmap('PiYG')
# cmapp = [cmap1,cmap,cmap]

IND = ['a) SW CRE','b) LW CRE','c) Net CRE']
IND1 = ['d)','e)','f)']
IND2 = ['g)','h)','i)']

fig, ax = plt.subplots(3,3, subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(28,15))
ax=ax.flatten()
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.05, right=0.55,
                    wspace=0.15, hspace=0.27)
  
for idc in np.arange(0,3):  
    cs=ax[idc].contourf(LONC, LATC, SAI[idc],clev1,
        transform = ccrs.PlateCarree(), cmap= 'bwr' ,extend='both')
    ax[idc].coastlines(linewidth=4)
    ax[idc].set_title(IND[idc],fontweight = 'bold',fontsize=20)
    ax[idc].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc].yaxis.set_major_formatter(lat_formatter)
    ax[idc].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc].axes.get_xaxis().set_ticklabels
    ax[idc].axes.get_yaxis().set_ticklabels
    ax[idc].axes.axis('tight')
    fig.text(0.02,0.70, 'ARISE',fontweight = 'bold',fontsize=20, rotation=90)
    
    ###################################################################
    ###################################################################
    cs=ax[idc+3].contourf(LONC, LATC, SSP245[idc],clev1,
        transform = ccrs.PlateCarree(), 
        cmap= 'bwr' ,extend='both')
    ax[idc+3].coastlines(linewidth=4)
    ax[idc+3].set_title(IND1[idc],fontweight = 'bold',fontsize=20)
    ax[idc+3].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc+3].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc+3].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc+3].yaxis.set_major_formatter(lat_formatter)
    ax[idc+3].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc+3].axes.get_xaxis().set_ticklabels
    ax[idc+3].axes.get_yaxis().set_ticklabels
    ax[idc+3].axes.axis('tight')
    fig.text(0.02,0.40, 'SSP2-45',fontweight = 'bold',fontsize=20, rotation=90)
    
    ######################################################################
    ######################################################################
    
    css=ax[idc+6].contourf(LONC, LATC, BIAS[idc],clevk,
        transform = ccrs.PlateCarree(), 
        cmap= 'bwr' ,extend='both')
    ax[idc+6].coastlines(linewidth=4)
    ax[idc+6].set_title(IND2[idc],fontweight = 'bold',fontsize=20)
    ax[idc+6].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc+6].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc+6].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc+6].yaxis.set_major_formatter(lat_formatter)
    ax[idc+6].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc+6].axes.get_xaxis().set_ticklabels
    ax[idc+6].axes.get_yaxis().set_ticklabels
    ax[idc+6].axes.axis('tight')
    # cbar=fig.colorbar(cs,  ax = ax[idc+6],orientation='vertical')
    # cbar.set_label('CRE (W/m²)', fontsize=12) 
    # cbar.ax.tick_params(labelsize=12)
    
    ax[idc+6].add_patch(Rectangle((-12, 7),
                      18, 6,
                      fc ='none', 
                      color = 'red',
                      linewidth =4,
                      linestyle = '-'))
    
    ax[idc+6].add_patch(Rectangle((-8, 16),
                      12, 6,
                      fc ='none', 
                      color = 'black',
                      linewidth =4,
                      linestyle = '-'))
    
        
    ax[idc+6].add_patch(Rectangle((10, -2),
                      16, 7,
                      fc ='none', 
                      color = 'green',
                      linewidth =4,
                      linestyle = '-'))

    fig.text(0.02,0.05,'ARISE-SSP2-4.5',fontweight = 'bold',fontsize=20, rotation=90)

## Draw the colorbar
cbar_ax = fig.add_axes([0.56, 0.42, 0.006, 0.4]) 
cbar=fig.colorbar(cs, cbar_ax, ax = ax[2] , orientation='vertical')
cbar.set_label('W/m²', fontsize=20,fontweight = 'bold') 
cbar.ax.tick_params(labelsize=17)

## Draw the colorbar
cbar_ax1 = fig.add_axes([0.56, 0.03, 0.006, 0.2]) 
cbar=fig.colorbar(css, cbar_ax1, ax = ax[8] , orientation='vertical')
cbar.set_label('W/m²', fontsize=20,fontweight = 'bold') 
cbar.ax.tick_params(labelsize=17)

xxxxx
plt.savefig("D:/SAI_Data/Comparison_SSP245&SAI-2.png", dpi=700, bbox_inches='tight')


###############################################################################
###############################################################################


LONC, LATC = np.meshgrid(lon,lat)

clev1 = np.arange(-120,130,10)
clevp = np.arange(-15, 16, 1)
cmap = plt.get_cmap('seismic') #coolwarm_r viridis_r
# cmap1 = plt.get_cmap('PiYG')
# cmapp = [cmap1,cmap,cmap]

IND = ['a) SW CRE','b) LW CRE','c) Net CRE']
IND1 = ['d)','e)','f)']
IND2 = ['g)','h)','i)']

fig, ax = plt.subplots(3,3, subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(28,15))
ax=ax.flatten()
fig.subplots_adjust(bottom=0.0, top=0.85, left=0.05, right=0.55,
                    wspace=0.15, hspace=0.27)
  
for idc in np.arange(0,3):  
    cs=ax[idc].contourf(LONC, LATC, SAI[idc],clev1,
        transform = ccrs.PlateCarree(), cmap= 'bwr' ,extend='both')
    ax[idc].coastlines(linewidth=4)
    ax[idc].set_title(IND[idc],fontweight = 'bold',fontsize=20)
    ax[idc].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc].yaxis.set_major_formatter(lat_formatter)
    ax[idc].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc].axes.get_xaxis().set_ticklabels
    ax[idc].axes.get_yaxis().set_ticklabels
    ax[idc].axes.axis('tight')
    fig.text(0.02,0.70, 'ARISE',fontweight = 'bold',fontsize=20, rotation=90)
    
    ###################################################################
    ###################################################################
    cs=ax[idc+3].contourf(LONC, LATC, SSP245[idc],clev1,
        transform = ccrs.PlateCarree(), 
        cmap= 'bwr' ,extend='both')
    ax[idc+3].coastlines(linewidth=4)
    ax[idc+3].set_title(IND1[idc],fontweight = 'bold',fontsize=20)
    ax[idc+3].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc+3].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc+3].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc+3].yaxis.set_major_formatter(lat_formatter)
    ax[idc+3].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc+3].axes.get_xaxis().set_ticklabels
    ax[idc+3].axes.get_yaxis().set_ticklabels
    ax[idc+3].axes.axis('tight')
    fig.text(0.02,0.40, 'SSP2-45',fontweight = 'bold',fontsize=20, rotation=90)
    
    ######################################################################
    ######################################################################
    
    css=ax[idc+6].contourf(LONC, LATC, 100*Perc_BIAS[idc].squeeze(),clevp,
        transform = ccrs.PlateCarree(), 
        cmap= 'bwr' ,extend='both')
    ax[idc+6].coastlines(linewidth=4)
    ax[idc+6].set_title(IND2[idc],fontweight = 'bold',fontsize=20)
    ax[idc+6].set_xticks(np.arange(-20,40,8), crs=ccrs.PlateCarree())
    lon_formatter = cticker.LongitudeFormatter()
    ax[idc+6].xaxis.set_major_formatter(lon_formatter)
    # Define the yticks for latitude
    ax[idc+6].set_yticks(np.arange(-10,30,4), crs=ccrs.PlateCarree())
    lat_formatter = cticker.LatitudeFormatter()
    ax[idc+6].yaxis.set_major_formatter(lat_formatter)
    ax[idc+6].add_feature(cfeature.BORDERS, linestyle='dashed',linewidth=1.5)   
    ax[idc+6].axes.get_xaxis().set_ticklabels
    ax[idc+6].axes.get_yaxis().set_ticklabels
    ax[idc+6].axes.axis('tight')
    # cbar=fig.colorbar(cs,  ax = ax[idc+6],orientation='vertical')
    # cbar.set_label('CRE (W/m²)', fontsize=12) 
    # cbar.ax.tick_params(labelsize=12) 
    ax[idc+6].add_patch(Rectangle((-12, 7),
                      18, 6,
                      fc ='none', 
                      color = 'red',
                      linewidth =4,
                      linestyle = '-'))
    
    ax[idc+6].add_patch(Rectangle((-8, 16),
                      12, 6,
                      fc ='none', 
                      color = 'black',
                      linewidth = 4,
                      linestyle = '-'))
    
    ax[idc+6].add_patch(Rectangle((10, -2),
                      16, 7,
                      fc ='none', 
                      color = 'green',
                      linewidth =4,
                      linestyle = '-'))
    fig.text(0.02,0.05,'ARISE-SSP2-4.5',fontweight = 'bold',fontsize=20, rotation=90)

## Draw the colorbar
cbar_ax = fig.add_axes([0.56, 0.42, 0.006, 0.4]) 
cbar=fig.colorbar(cs, cbar_ax, ax = ax[2] , orientation='vertical')
cbar.set_label('W/m²', fontsize=20,fontweight = 'bold') 
cbar.ax.tick_params(labelsize=17)

## Draw the colorbar
cbar_ax1 = fig.add_axes([0.56, 0.03, 0.006, 0.2]) 
cbar=fig.colorbar(css, cbar_ax1, ax = ax[8] , orientation='vertical')
cbar.set_label('%', fontsize=20,fontweight = 'bold') 
cbar.ax.tick_params(labelsize=17)

xxxxxxxx
plt.savefig("D:/SAI_Data/Comparison_SSP245&SAI_perc_2.png", dpi=700, bbox_inches='tight')


