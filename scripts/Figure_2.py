# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:02:03 2024

@author: ad7gb
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Rectangle




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

###############################################################################
###############################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lwR = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwR_sd = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lwB = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwB_sd = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lwCA = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SAI_lwCA_sd = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 

################# global
weights = np.cos(np.deg2rad(SAI_lw_net.lat))
weights.name = "weights"  
SAI_lw_weighted = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'))['FLNS'].weighted(weights)
SAI_lwG  = SAI_lw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SAI_lwG_sd  = SAI_lw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SAI_lw_weighted = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'), lat=lat)['FLNS'].weighted(weights)
SAI_lwTRP = SAI_lw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values
SAI_lwTRP_sd = SAI_lw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values


#SAI_lwTRPP = SAI_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 


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

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_clrR = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrR_sd = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_clrB = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrB_sd = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_clrCA = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SAI_lw_clrCA_sd = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SAI_lw_clrTRP = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 

################# global
weights = np.cos(np.deg2rad(SAI_lw_net_clr.lat))
weights.name = "weights"  
SAI_lwclr_weighted = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'))['FLNSC'].weighted(weights)
SAI_lw_clrG  = SAI_lwclr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values
SAI_lw_clrG_sd  = SAI_lwclr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SAI_lwclr_weighted = SAI_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'), lat=lat)['FLNSC'].weighted(weights)
SAI_lw_clrTRP = SAI_lwclr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values
SAI_lw_clrTRP_sd = SAI_lwclr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

#################################################################
############################################################################
SAI_CRE_lwR = -SAI_lwR+SAI_lw_clrR
SAI_CRE_lwB = -SAI_lwB+SAI_lw_clrB
SAI_CRE_lwCA = -SAI_lwCA+SAI_lw_clrCA
SAI_CRE_lwTRP = -SAI_lwTRP+SAI_lw_clrTRP
SAI_CRE_lwG = -SAI_lwG+SAI_lw_clrG

#####################################################################
###########################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNS/'
ddirsR = os.listdir(fsai)

SAI_sw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNS.203501-206912.nc')
SAI_sw_net.coords['lon'] = (SAI_sw_net.coords['lon'] + 180) % 360 - 180
SAI_sw_net = SAI_sw_net.sortby(SAI_sw_net.lon)

laC  = SAI_sw_net.lat.values
loC  = SAI_sw_net.lon.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_swR = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swR_sd = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_swB = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swB_sd = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_swCA = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SAI_swCA_sd = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SAI_swTRP = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 

################# global
weights = np.cos(np.deg2rad(SAI_sw_net.lat))
weights.name = "weights"  
SAI_sw_weighted = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'))['FSNS'].weighted(weights)
SAI_swG  = SAI_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values
SAI_swG_sd  = SAI_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values  


#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SAI_sw_weighted = SAI_sw_net.sel(time = slice('2035-02-01', '2069-12-31'), lat=lat)['FSNS'].weighted(weights)
SAI_swTRP = SAI_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values
SAI_swTRP_sd = SAI_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

####################################################################################
#######################################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNSC/'
ddirsR = os.listdir(fsai)

SAI_sw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNSC.203501-206912.nc')
SAI_sw_net_clr.coords['lon'] = (SAI_sw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_sw_net_clr = SAI_sw_net_clr.sortby(SAI_sw_net_clr.lon)

laC  = SAI_sw_net_clr.lat.values
loC  = SAI_sw_net_clr.lon.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_clrR = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrR_sd = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_clrB = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrB_sd = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_clrCA = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SAI_sw_clrCA_sd = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SAI_sw_clrTRP = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 

################# global
weights = np.cos(np.deg2rad(SAI_sw_net_clr.lat))
weights.name = "weights"  
SAI_sw_clr_weighted = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'))['FSNSC'].weighted(weights)
SAI_sw_clrG  = SAI_sw_clr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SAI_sw_clrG_sd  = SAI_sw_clr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SAI_sw_clr_weighted = SAI_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'), lat=lat)['FSNSC'].weighted(weights)
SAI_sw_clrTRP = SAI_sw_clr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SAI_sw_clrTRP_sd = SAI_sw_clr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

##################################################
################################################

SAI_CRE_swR = SAI_swR-SAI_sw_clrR
SAI_CRE_swB = SAI_swB-SAI_sw_clrB
SAI_CRE_swCA = SAI_swCA-SAI_sw_clrCA
SAI_CRE_swTRP = SAI_swTRP-SAI_sw_clrTRP
SAI_CRE_swG = SAI_swG-SAI_sw_clrG

SAI_CRE_R = SAI_CRE_swR+SAI_CRE_lwR
SAI_CRE_B = SAI_CRE_swB+SAI_CRE_lwB
SAI_CRE_CA = SAI_CRE_swCA+SAI_CRE_lwCA
SAI_CRE_TRP = SAI_CRE_swTRP+SAI_CRE_lwTRP
SAI_CRE_G = SAI_CRE_swG+SAI_CRE_lwG

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

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lwR = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwR_sd = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lwB = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwB_sd = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lwCA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 
SSP245_lwCA_sd = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNS'].values

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SSP245_lwTRP = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNS'].values 

################# global
weights = np.cos(np.deg2rad(SSP245_lw_net.lat))
weights.name = "weights"  
SSP245_lw_weighted = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'))['FLNS'].weighted(weights)
SSP245_lwG  = SSP245_lw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_lwG_sd  = SSP245_lw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SSP245_lw_weighted = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'), lat =lat)['FLNS'].weighted(weights)
SSP245_lwTRP = SSP245_lw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_lwTRP_sd = SSP245_lw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 


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

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_clrR = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrR_sd = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_clrB = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrB_sd = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values  

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_clrCA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 
SSP245_lw_clrCA_sd = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FLNSC'].values 

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SSP245_lw_clrTRP = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FLNSC'].values 

################# global
weights = np.cos(np.deg2rad(SSP245_lw_net_clr.lat))
weights.name = "weights"  
SSP245_lw_clr_weighted = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'))['FLNSC'].weighted(weights)
SSP245_lw_clrG  = SSP245_lw_clr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_lw_clrG_sd  = SSP245_lw_clr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SSP245_lw_clr_weighted = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat=lat)['FLNSC'].weighted(weights)
SSP245_lw_clrTRP = SSP245_lw_clr_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_lw_clrTRP_sd = SSP245_lw_clr_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

###############################################################
##########################################################################
SSP245_CRE_lwR = -SSP245_lwR +SSP245_lw_clrR
SSP245_CRE_lwB = -SSP245_lwB +SSP245_lw_clrB
SSP245_CRE_lwCA = -SSP245_lwCA+SSP245_lw_clrCA
SSP245_CRE_lwTRP = -SSP245_lwTRP+SSP245_lw_clrTRP
SSP245_CRE_lwG = -SSP245_lwG +SSP245_lw_clrG


#####################################################################
###########################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNS/'
ddirs  = os.listdir(fssp)

SSP245_sw_net = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FSNS.201501-206912.nc')

SSP245_sw_net.coords['lon'] = (SSP245_sw_net.coords['lon'] + 180) % 360 - 180
SSP245_sw_net = SSP245_sw_net.sortby(SSP245_sw_net.lon)

laC  = SSP245_sw_net.lat.values
loC  = SSP245_sw_net.lon.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_swR = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swR_sd = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_swB = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swB_sd = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_swCA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 
SSP245_swCA_sd = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNS'].values 

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SSP245_swTRP = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNS'].values 

################# global
weights = np.cos(np.deg2rad(SSP245_sw_net.lat))
weights.name = "weights"  
SSP245_sw_weighted = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'))['FSNS'].weighted(weights)
SSP245_swG  = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_swG_sd  = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SSP245_sw_weighted = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'), lat = lat)['FSNS'].weighted(weights)
SSP245_swTRP = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_swTRP_sd = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNSC/'
ddirs  = os.listdir(fssp)

SSP245_sw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FSNSC.201501-206912.nc')

SSP245_sw_net_clr.coords['lon'] = (SSP245_sw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_sw_net_clr = SSP245_sw_net_clr.sortby(SSP245_sw_net_clr.lon)

laC  = SSP245_sw_net_clr.lat.values
loC  = SSP245_sw_net_clr.lon.values

####################################################################
########################################################################
##################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_clrR = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrR_sd = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_clrB = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrB_sd = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_clrCA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 
SSP245_sw_clrCA_sd = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat')).groupby('time.year').mean(dim ='time')['FSNSC'].values 

# #################### yellow box 
# lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
# SSP245_sw_clrTRP = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat).mean(('lon','lat')).groupby('time.month').mean(dim ='time')['FSNSC'].values 

################# global
weights = np.cos(np.deg2rad(SSP245_sw_net_clr.lat))
weights.name = "weights"  
SSP245_sw_weighted = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'))['FSNSC'].weighted(weights)
SSP245_sw_clrG  = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_sw_clrG_sd  = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 

#################### yellow box 
lat = laC[np.where((laC>=-14)&(laC<=14))[0]]
SSP245_sw_weighted = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat=lat)['FSNSC'].weighted(weights)
SSP245_sw_clrTRP = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.month').mean(dim ='time').values 
SSP245_sw_clrTRP_sd = SSP245_sw_weighted.mean(('lon','lat')).groupby('time.year').mean(dim ='time').values 





SSP245_CRE_swR  = SSP245_swR- SSP245_sw_clrR
SSP245_CRE_swB  = SSP245_swB- SSP245_sw_clrB
SSP245_CRE_swCA = SSP245_swCA- SSP245_sw_clrCA
SSP245_CRE_swTRP = SSP245_swTRP- SSP245_sw_clrTRP
SSP245_CRE_swG  = SSP245_swG- SSP245_sw_clrG

SSP245_CRE_R = SSP245_CRE_swR + SSP245_CRE_lwR
SSP245_CRE_B = SSP245_CRE_swB + SSP245_CRE_lwB
SSP245_CRE_CA = SSP245_CRE_swCA + SSP245_CRE_lwCA
SSP245_CRE_TRP = SSP245_CRE_swTRP + SSP245_CRE_lwTRP
SSP245_CRE_G = SSP245_CRE_swG + SSP245_CRE_lwG

###############################################################################
###############################################################################
############## red box
Bias_net_R = SAI_CRE_R-SSP245_CRE_R
Bias_net_swR = SAI_CRE_swR-SSP245_CRE_swR
Bias_net_lwR = SAI_CRE_lwR-SSP245_CRE_lwR

Perc_Bias_net_R = np.array([Bias_net_R.mean()/abs(SSP245_CRE_R.mean()) if SSP245_CRE_R.mean() !=0 else 0])
Perc_Bias_net_sw_R = np.array([Bias_net_swR.mean()/abs(SSP245_CRE_swR.mean()) if SSP245_CRE_swR.mean() !=0 else 0])
Perc_Bias_net_lw_R = np.array([Bias_net_lwR.mean()/abs(SSP245_CRE_lwR.mean()) if SSP245_CRE_lwR.mean() !=0 else 0])

######################## Black box
Bias_net_B = SAI_CRE_B-SSP245_CRE_B
Bias_net_swB = SAI_CRE_swB-SSP245_CRE_swB
Bias_net_lwB = SAI_CRE_lwB-SSP245_CRE_lwB

Perc_Bias_net_B  = np.array([Bias_net_B.mean()/abs(SSP245_CRE_B.mean()) if SSP245_CRE_B.mean() !=0 else 0])
Perc_Bias_net_sw_B  = np.array([Bias_net_swB.mean()/abs(SSP245_CRE_swB.mean() ) if SSP245_CRE_swB.mean() !=0 else 0])
Perc_Bias_net_lw_B  = np.array([Bias_net_lwB.mean()/abs(SSP245_CRE_lwB.mean() ) if SSP245_CRE_lwB.mean() !=0 else 0])


######################## green box
Bias_net_CA = SAI_CRE_CA-SSP245_CRE_CA
Bias_net_lwCA = SAI_CRE_lwCA-SSP245_CRE_lwCA
Bias_net_swCA = SAI_CRE_swCA-SSP245_CRE_swCA

Perc_Bias_net_CA  = np.array([Bias_net_CA.mean()/abs(SSP245_CRE_CA.mean()) if SSP245_CRE_CA.mean() !=0 else 0])
Perc_Bias_net_sw_CA  = np.array([Bias_net_swCA.mean()/abs(SSP245_CRE_swCA.mean() ) if SSP245_CRE_swCA.mean() !=0 else 0])
Perc_Bias_net_lw_CA  = np.array([Bias_net_lwCA.mean()/abs(SSP245_CRE_lwCA.mean() ) if SSP245_CRE_lwCA.mean() !=0 else 0])

######################## yellow box
Bias_net_TRP = SAI_CRE_TRP-SSP245_CRE_TRP
Bias_net_swTRP = SAI_CRE_swTRP-SSP245_CRE_swTRP
Bias_net_lwTRP = SAI_CRE_lwTRP-SSP245_CRE_lwTRP

Perc_Bias_net_TRP  = np.array([Bias_net_TRP.mean()/abs(SSP245_CRE_TRP.mean()) if SSP245_CRE_TRP.mean() !=0 else 0])
Perc_Bias_net_sw_TRP  = np.array([Bias_net_swTRP.mean()/abs(SSP245_CRE_swTRP.mean() ) if SSP245_CRE_swTRP.mean() !=0 else 0])
Perc_Bias_net_lw_TRP  = np.array([Bias_net_lwTRP.mean()/abs(SSP245_CRE_lwTRP.mean()) if SSP245_CRE_lwTRP.mean() !=0 else 0])

#####################################################
#####################################################
######## global
Bias_net_G = SAI_CRE_G-SSP245_CRE_G
Bias_net_swG = SAI_CRE_swG - SSP245_CRE_swG
Bias_net_lwG = SAI_CRE_lwG - SSP245_CRE_lwG

Perc_Bias_net_G = np.array([Bias_net_G.mean()/abs(SSP245_CRE_G.mean()) if SSP245_CRE_G.mean() !=0 else 0])
Perc_Bias_net_sw_G = np.array([Bias_net_swG.mean()/abs(SSP245_CRE_swG.mean()) if SSP245_CRE_swG.mean() !=0 else 0])
Perc_Bias_net_lw_G = np.array([Bias_net_lwG.mean()/abs(SSP245_CRE_lwG.mean()) if SSP245_CRE_lwG.mean() !=0 else 0])

#####################################################
#####################################################
Perc_BIAS_R = np.array([100*Perc_Bias_net_sw_R[0],100*Perc_Bias_net_lw_R[0],100*Perc_Bias_net_R[0]])
Perc_BIAS_B = np.array([100*Perc_Bias_net_sw_B[0],100*Perc_Bias_net_lw_B[0],100*Perc_Bias_net_B[0]])
Perc_BIAS_CA = np.array([100*Perc_Bias_net_sw_CA[0],100*Perc_Bias_net_lw_CA[0],100*Perc_Bias_net_CA[0]])
Perc_BIAS_TRP = np.array([100*Perc_Bias_net_sw_TRP[0],100*Perc_Bias_net_lw_TRP[0],100*Perc_Bias_net_TRP[0]])
Perc_BIAS_G = np.array([100*Perc_Bias_net_sw_G[0],100*Perc_Bias_net_lw_G[0],100*Perc_Bias_net_G[0]])


BIAS_R = np.array([Bias_net_swR.mean(),Bias_net_lwR.mean(),Bias_net_R.mean()])
BIAS_B = np.array([Bias_net_swB.mean(),Bias_net_lwB.mean(),Bias_net_B.mean()])
BIAS_CA = np.array([Bias_net_swCA.mean(),Bias_net_lwCA.mean(),Bias_net_CA.mean()])
BIAS_TRP = np.array([Bias_net_swTRP.mean(),Bias_net_lwTRP.mean(),Bias_net_TRP.mean()])
BIAS_G = np.array([Bias_net_swG.mean(),Bias_net_lwG.mean(),Bias_net_G.mean()])

################################################################################
########### calculate the monthly standard deviation of the difference between SAI CRE and SSP2-4.5 CRE
################################################################################
################ SWA

#### longwave CRE
SAI_net_lwR = -SAI_lwR_sd+SAI_lw_clrR_sd
#### shortwave
SAI_net_swR = SAI_swR_sd-SAI_sw_clrR_sd
### net (longwave +shortwave) CRE
SAI_net = SAI_net_swR+SAI_net_lwR
############################### SSP2-4.5
#### longwave CRE
SSP245_net_lwR = -SSP245_lwR_sd+SSP245_lw_clrR_sd
#### shortwave CRE
SSP245_net_swR = SSP245_swR_sd-SSP245_sw_clrR_sd
### net (longwave +shortwave) CRE
SSP245_net = SSP245_net_swR+SSP245_net_lwR
################## difference between SAI and SSP245
####### longwave CRE
diff_lwR = SAI_net_lwR-SSP245_net_lwR
std_diff_lwR =np.std(diff_lwR)
####### shortwave CRE
diff_swR = SAI_net_swR-SSP245_net_swR
std_diff_swR = np.std(diff_swR)
####### net CRE
diff_netR = SAI_net-SSP245_net
std_diff_netR =np.std(diff_netR)
    
################ Sahara Region
############################### ARISE-SAI
#### longwave CRE
SAI_net_lwB = -SAI_lwB_sd+SAI_lw_clrB_sd
#### shortwave
SAI_net_swB = SAI_swB_sd-SAI_sw_clrB_sd
### net (longwave +shortwave) CRE
SAI_net = SAI_net_swB+SAI_net_lwB
############################### SSP2-4.5
#### longwave CRE
SSP245_net_lwB = -SSP245_lwB_sd+SSP245_lw_clrB_sd
#### shortwave CRE
SSP245_net_swB = SSP245_swB_sd-SSP245_sw_clrB_sd
### net (longwave +shortwave) CRE
SSP245_net = SSP245_net_swB+SSP245_net_lwB
################## difference between SAI and SSP245
####### longwave CRE
diff_lwB = SAI_net_lwB-SSP245_net_lwB
std_diff_lwB =np.std(diff_lwB)
####### shortwave CRE
diff_swB = SAI_net_swB-SSP245_net_swB
std_diff_swB = np.std(diff_swB)
####### net CRE
diff_netB = SAI_net-SSP245_net
std_diff_netB = np.std(diff_netB)

################ CA
############################### ARISE-SAI
#### longwave CRE
SAI_net_lwCA = -SAI_lwCA_sd+SAI_lw_clrCA_sd
#### shortwave
SAI_net_swCA = SAI_swCA_sd-SAI_sw_clrCA_sd
### net (longwave +shortwave) CRE
SAI_net = SAI_net_swCA+SAI_net_lwCA
############################### SSP2-4.5
#### longwave CRE
SSP245_net_lwCA = -SSP245_lwCA_sd+SSP245_lw_clrCA_sd
#### shortwave CRE
SSP245_net_swCA = SSP245_swCA_sd-SSP245_sw_clrCA_sd
### net (longwave +shortwave) CRE
SSP245_net = SSP245_net_swCA+SSP245_net_lwCA
################## difference between SAI and SSP245
####### longwave CRE
diff_lwCA = SAI_net_lwCA-SSP245_net_lwCA
std_diff_lwCA =np.std(diff_lwCA)
####### shortwave CRE
diff_swCA = SAI_net_swCA-SSP245_net_swCA
std_diff_swCA= np.std(diff_swCA)
####### net CRE
diff_netCA = SAI_net-SSP245_net
std_diff_netCA = np.std(diff_netCA)
    

############### interannual standard deviation 
################ globe
############################### ARISE-SAI
#### longwave CRE
SAI_net_lwG = -SAI_lwG_sd+SAI_lw_clrG_sd
#### shortwave
SAI_net_swG = SAI_swG_sd-SAI_sw_clrG_sd
### net (longwave +shortwave) CRE
SAI_net = SAI_net_swG+SAI_net_lwG
############################### SSP2-4.5
#### longwave CRE
SSP245_net_lwG = -SSP245_lwG_sd+SSP245_lw_clrG_sd
#### shortwave CRE
SSP245_net_swG = SSP245_swG_sd-SSP245_sw_clrG_sd
### net (longwave +shortwave) CRE
SSP245_net = SSP245_net_swG+SSP245_net_lwG
################## difference between SAI and SSP245
####### longwave CRE
diff_lwG = SAI_net_lwG-SSP245_net_lwG
std_diff_lwG =np.std(diff_lwG)
####### shortwave CRE
diff_swG = SAI_net_swG-SSP245_net_swG
std_diff_swG= np.std(diff_swG)
####### net CRE
diff_netG = SAI_net-SSP245_net
std_diff_netG = np.std(diff_netG)


################ tropical
############################### ARISE-SAI
#### longwave CRE
SAI_net_lwTRP = -SAI_lwTRP_sd+SAI_lw_clrTRP_sd
#### shortwave
SAI_net_swTRP = SAI_swTRP_sd-SAI_sw_clrTRP_sd
### net (longwave +shortwave) CRE
SAI_net = SAI_net_swTRP+SAI_net_lwTRP
############################### SSP2-4.5
#### longwave CRE
SSP245_net_lwTRP = -SSP245_lwTRP_sd+SSP245_lw_clrTRP_sd
#### shortwave CRE
SSP245_net_swTRP = SSP245_swTRP_sd-SSP245_sw_clrTRP_sd
### net (longwave +shortwave) CRE
SSP245_net = SSP245_net_swTRP+SSP245_net_lwTRP
################## difference between SAI and SSP245
####### longwave CRE
diff_lwTRP = SAI_net_lwTRP-SSP245_net_lwTRP
std_diff_lwTRP =np.std(diff_lwTRP)
####### shortwave CRE
diff_swTRP = SAI_net_swTRP-SSP245_net_swTRP
std_diff_swTRP= np.std(diff_swTRP)
####### net CRE
diff_netTRP = SAI_net-SSP245_net
std_diff_netTRP = np.std(diff_netTRP)



std_diff_R = np.array([np.std(diff_swR), np.std(diff_lwR), np.std(diff_netR)])
std_diff_B = np.array([np.std(diff_swB), np.std(diff_lwB), np.std(diff_netB)])
std_diff_CA = np.array([np.std(diff_swCA), np.std(diff_lwCA), np.std(diff_netCA)])
std_diff_TRP = np.array([np.std(diff_swTRP), np.std(diff_lwTRP), np.std(diff_netTRP)])
std_diff_G = np.array([np.std(diff_swG), np.std(diff_lwG), np.std(diff_netG)])


###############################################################################
###############################################################################


colors = ['Black','Green','blue'] #,'blue'
colors1 = ['Red','Black','Green','darkviolet','blue'] #,'blue'
CRE = ['SW CRE', 'LW CRE', 'NET CRE']

MTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
width = 0.14

N = np.arange(len(MTH))

fig, ax = plt.subplots(2,2, figsize=(17,12), tight_layout=False)

ax[0,0].bar(N-0.14, Bias_net_swR, width, color = colors[0])
ax[0,0].bar(N, Bias_net_lwR, width, color = colors[1])
ax[0,0].bar(N+0.14, Bias_net_R, width, color = colors[2])


ax[0,0].grid()
ax[0,0].set_xticks(N)
ax[0,0].set_xticklabels('')
ax[0,0].set_title('a) SWA', fontweight='bold', fontsize=20)
ax[0,0].set_ylabel('ΔCRE (W/m²)', fontweight='bold', fontsize=20)

ax[0,0].legend(('SW','LW','NET'), loc='upper left',
                  fancybox=True,  shadow=False, ncol=3,prop={'size':15})


ax[0,1].bar(N-0.14, Bias_net_swB, width, color = colors[0])
ax[0,1].bar(N, Bias_net_lwB, width, color = colors[1])
ax[0,1].bar(N+0.14, Bias_net_B, width, color = colors[2])

ax[0,1].grid()
ax[0,1].set_xticks(N)
ax[0,1].set_xticklabels(MTH, fontweight='bold', fontsize=20, rotation=30)
ax[0,1].set_title('b) SAH', fontweight='bold', fontsize=20)
ax[0,1].set_ylabel('ΔCRE (W/m²)', fontweight='bold', fontsize=20)

ax[1,0].bar(N-0.14, Bias_net_swCA, width, color = colors[0])
ax[1,0].bar(N, Bias_net_lwCA, width, color = colors[1])
ax[1,0].bar(N+0.14, Bias_net_CA, width, color = colors[2])

ax[1,0].grid()
#ax[2].set_yticks([-2,-1,0,1,2,3,4,5,6,7,8]) 
ax[1,0].set_title('c) CA', fontweight='bold', fontsize=20)
ax[1,0].set_xticks(N)
ax[1,0].set_xticklabels(MTH, fontweight='bold', fontsize=20, rotation=30)
ax[1,0].set_ylabel('ΔCRE (W/m²)', fontweight='bold', fontsize=20)
ax[0,0].set_ylim([-3 , 2.5])
ax[0,1].set_ylim([-3 , 2.5])
ax[1,0].set_ylim([-3 , 2.5])


N = np.arange(len(CRE))
ax[1,1].bar(N-0.28, BIAS_R, width, color = colors1[0], label='SWA')
ax[1,1].bar(N-0.14, BIAS_B, width, color = colors1[1], label='SAH')
ax[1,1].bar(N, BIAS_CA, width, color = colors1[2], label='CA')
ax[1,1].bar(N+0.14, BIAS_TRP, width, color = colors1[3], label='Tropics')
ax[1,1].bar(N+0.28, BIAS_G, width, color = colors1[4], label='Global')

ax[1,1].errorbar(N-0.28, BIAS_R, std_diff_R,fmt='.', linewidth=2, capsize=0,color = colors[0])
ax[1,1].errorbar(N-0.14, BIAS_B, std_diff_B,fmt='.', linewidth=2, capsize=0,color = colors[0])
ax[1,1].errorbar(N, BIAS_CA, std_diff_CA,fmt='.', linewidth=2, capsize=0,color = colors[0])
ax[1,1].errorbar(N+0.14, BIAS_TRP, std_diff_TRP,fmt='.', linewidth=2, capsize=0,color = colors[0])
ax[1,1].errorbar(N+0.28, BIAS_G, std_diff_G,fmt='.', linewidth=2, capsize=0,color = colors[0])


ax[1,1].grid(True, linestyle='-.')

ax[1,1].set_ylim([-1.5, 2]) 
ax[1,1].set_yticks([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5])  
ax[1,1].set_xticks(N)
ax[1,1].set_title('d) Average', fontweight='bold', fontsize=20)
# Adding labels to the bars
for i in range(3):
    ax[1,1].grid(True, linestyle='-.')
    if i==1: 
        ax[1,1].text(i-0.28, BIAS_R[i] , '+'+'{:.2f}'.format(Perc_BIAS_R[i])+'%', fontweight = 'bold', 
                ha='center', va='bottom', color = colors1[0])
        ax[1,1].text(i-0.14, BIAS_B[i], '+'+'{:.2f}'.format(abs(Perc_BIAS_B[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[1])
        ax[1,1].text(i, BIAS_CA[i], '+'+'{:.2f}'.format(abs(Perc_BIAS_CA[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[2])
        ax[1,1].text(i+0.14, BIAS_TRP[i], '+'+'{:.2f}'.format(abs(Perc_BIAS_TRP[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[3])
        ax[1,1].text(i+0.28, BIAS_G[i] , '+'+'{:.2f}'.format(Perc_BIAS_G[i])+'%', fontweight = 'bold', 
                ha='center', va='bottom', color = colors1[4])  
    elif i==2:
        ax[1,1].text(i-0.28, BIAS_R[i]-0.2 , '+'+'{:.2f}'.format(abs(Perc_BIAS_R[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[0])
        ax[1,1].text(i-0.14, BIAS_B[i], '-'+'{:.2f}'.format(abs(Perc_BIAS_B[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[1])
        ax[1,1].text(i, BIAS_CA[i], '-'+'{:.2f}'.format(abs(Perc_BIAS_CA[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[2])
        ax[1,1].text(i+0.14, BIAS_TRP[i], '-'+'{:.2f}'.format(abs(Perc_BIAS_TRP[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[3])
        ax[1,1].text(i+0.28, BIAS_G[i] , '{:.2f}'.format(-Perc_BIAS_G[i])+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[4]) 
    else:
        ax[1,1].text(i-0.28, BIAS_R[i]-0.2 , '+'+'{:.2f}'.format(abs(Perc_BIAS_R[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[0])
        ax[1,1].text(i-0.14, BIAS_B[i]-0.2, '+'+'{:.2f}'.format(abs(Perc_BIAS_B[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[1])
        ax[1,1].text(i, BIAS_CA[i]-0.2, '+'+'{:.2f}'.format(abs(Perc_BIAS_CA[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[2])
        ax[1,1].text(i+0.14, BIAS_TRP[i], '-'+'{:.2f}'.format(abs(Perc_BIAS_TRP[i]))+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[3])
        ax[1,1].text(i+0.28, BIAS_G[i] , '{:.2f}'.format(-Perc_BIAS_G[i])+'%', fontweight = 'bold',
                ha='center', va='bottom', color = colors1[4])
        
#ax[3].set_yticks([-20, -11, -10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,20,40,60,80,100,120,140])
ax[1,1].set_xticklabels(CRE, fontweight='bold', fontsize=20)
ax[1,1].set_ylabel('ΔCRE (W/m²)', fontweight='bold', fontsize=20)

# ax[3].legend(['Southern West Africa','Sahara Region','Central Africa', 'Global'], loc= 'lower center',
#           shadow = False, ncol = 1, prop={'size':15}) #.get_frame().set_facecolor('b')

ax[1,1].legend(('Southern West Africa','Sahara Region','Central Africa', 'Tropics','Global'), 
          loc='upper center',bbox_to_anchor=(0.5,-0.1), fancybox=True,  shadow=False, 
          ncol=3,prop={'size':15})

ax[1,1].plot([0, 2], [0 ,0] ,'k--',linewidth=1.5)


plt.savefig("D:/SAI_Data/Figure_2.pdf", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_2.png", dpi=700, bbox_inches='tight')


