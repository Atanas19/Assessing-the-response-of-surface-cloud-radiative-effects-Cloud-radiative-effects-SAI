# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:00:24 2024

@author: ad7gb
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import scipy.interpolate as inter 
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind

###############################################################################
###############################################################################
########################### SSP2-4.5 #############################################
###############################################################################
###############################################################################


f = 'D:/SAI_Data/SSP245/CLDTOT/'
ddirsM = os.listdir(f)

#################################################################################
#################################################################################
###################### Total cloud
Tot_cld = xr.open_mfdataset(f+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.CLDTOT.201501-206912.nc')

Tot_cld.coords['lon'] = (Tot_cld.coords['lon'] + 180) % 360 - 180
Tot_cld = Tot_cld.sortby(Tot_cld.lon)
laC  = Tot_cld.lat.values
loC  = Tot_cld.lon.values

Tot_cld = Tot_cld.convert_calendar(calendar = 'gregorian', align_on = 'date')


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
Tot_cld_WA = Tot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
Tot_cld_SA = Tot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
Tot_cld_CA = Tot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values


#################################################################################
##############################################################################
############################## low cloud

f = 'D:/SAI_Data/SSP245/CLDLOW/'
ddirsM = os.listdir(f)
 
low_cld = xr.open_mfdataset(f+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.CLDLOW.201501-206912.nc')

low_cld.coords['lon'] = (low_cld.coords['lon'] + 180) % 360 - 180
low_cld = low_cld.sortby(low_cld.lon)
laC  = low_cld.lat.values
loC  = low_cld.lon.values


low_cld = low_cld.convert_calendar(calendar = 'gregorian', align_on = 'date')

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
low_cld_WA = low_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
low_cld_SA = low_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
low_cld_CA = low_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values


###############################################################################
###############################################################################
########################### ARISE #############################################
###############################################################################
###############################################################################

f = 'D:/SAI_Data/ARISE_WACCM/CLDTOT/'
ddirsM = os.listdir(f)

###############################################################################
###############################################################################
############################## Total cloud ####################################
###############################################################################
###############################################################################

ATot_cld = xr.open_mfdataset(f+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.CLDTOT.203501-206912.nc')

ATot_cld.coords['lon'] = (ATot_cld.coords['lon'] + 180) % 360 - 180
ATot_cld = ATot_cld.sortby(ATot_cld.lon)
laC  = ATot_cld.lat.values
loC  = ATot_cld.lon.values


ATot_cld = ATot_cld.convert_calendar(calendar = 'gregorian', align_on = 'date')


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
ATot_cld_WA = ATot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
ATot_cld_SA = ATot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
ATot_cld_CA = ATot_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDTOT'].values


#################################################################################
##############################################################################
############################## low cloud 

f = 'D:/SAI_Data/ARISE_WACCM/CLDLOW/'
ddirsM = os.listdir(f)

Alow_cld = xr.open_mfdataset(f+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.CLDLOW.203501-206912.nc')

Alow_cld.coords['lon'] = (Alow_cld.coords['lon'] + 180) % 360 - 180
Alow_cld = Alow_cld.sortby(Alow_cld.lon)
laC  = Alow_cld.lat.values
loC  = Alow_cld.lon.values

Alow_cld = Alow_cld.convert_calendar(calendar = 'gregorian', align_on = 'date')


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
Alow_cld_WA = Alow_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
Alow_cld_SA = Alow_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
Alow_cld_CA = Alow_cld.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['CLDLOW'].values


########################################################################################################
########################################################################################################

Bias_Tot_CA = 100*(ATot_cld_CA - Tot_cld_CA)
Bias_Tot_WA = 100*(ATot_cld_WA - Tot_cld_WA)
Bias_Tot_SA = 100*(ATot_cld_SA - Tot_cld_SA)


Bias_low_CA = 100*(Alow_cld_CA - low_cld_CA)
Bias_low_WA = 100*(Alow_cld_WA - low_cld_WA)
Bias_low_SA = 100*(Alow_cld_SA - low_cld_SA)




####################################################################################################
#####################################################################################################


fsai   = 'D:/SAI_Data/ARISE_WACCM/FLNS/'
ddirsR = os.listdir(fsai)

SAI_lw_net = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FLNS.203501-206912.nc')
SAI_lw_net.coords['lon'] = (SAI_lw_net.coords['lon'] + 180) % 360 - 180
SAI_lw_net = SAI_lw_net.sortby(SAI_lw_net.lon)

SAI_lw_net = SAI_lw_net.convert_calendar(calendar = 'gregorian', align_on = 'date')


laC = SAI_lw_net.lat.values
loC = SAI_lw_net.lon.values

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_WA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_SA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_CA = SAI_lw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

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


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_lw_clr_WA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_clr_SA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_clr_CA = SAI_lw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_WA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_SA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_CA = SAI_sw_net.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values


####################################################################################
#######################################################################################

fsai   = 'D:/SAI_Data/ARISE_WACCM/FSNSC/'
ddirsR = os.listdir(fsai)

SAI_sw_net_clr = xr.open_mfdataset(fsai+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.FSNSC.203501-206912.nc')

SAI_sw_net_clr.coords['lon'] = (SAI_sw_net_clr.coords['lon'] + 180) % 360 - 180
SAI_sw_net_clr = SAI_sw_net_clr.sortby(SAI_sw_net_clr.lon)
laC  = SAI_sw_net_clr.lat.values
loC  = SAI_sw_net_clr.lon.values

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SAI_sw_clr_WA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_clr_SA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_clr_CA = SAI_sw_net_clr.sel(time = slice('2035-02-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values


SAI_CRE_sw_CA = SAI_sw_CA-SAI_sw_clr_CA
SAI_CRE_sw_SA = SAI_sw_SA-SAI_sw_clr_SA
SAI_CRE_sw_WA = SAI_sw_WA-SAI_sw_clr_WA

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


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_WA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_SA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_CA = SSP245_lw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values


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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_lw_clr_WA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_clr_SA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_clr_CA = SSP245_lw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values


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

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_WA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_SA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_CA = SSP245_sw_net.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values


####################################################################################
#######################################################################################

fssp   = 'D:/SAI_Data/SSP245/FSNSC/'
ddirs  = os.listdir(fssp)

SSP245_sw_net_clr = xr.open_mfdataset(fssp+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.FSNSC.201501-206912.nc')

SSP245_sw_net_clr.coords['lon'] = (SSP245_sw_net_clr.coords['lon'] + 180) % 360 - 180
SSP245_sw_net_clr = SSP245_sw_net_clr.sortby(SSP245_sw_net_clr.lon)
laC  = SSP245_sw_net_clr.lat.values
loC  = SSP245_sw_net_clr.lon.values

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
SSP245_sw_clr_WA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_clr_SA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_clr_CA = SSP245_sw_net_clr.sel(time = slice('2035-02-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values


SSP245_CRE_sw_CA = SSP245_sw_CA- SSP245_sw_clr_CA
SSP245_CRE_sw_SA = SSP245_sw_SA- SSP245_sw_clr_SA
SSP245_CRE_sw_WA = SSP245_sw_WA- SSP245_sw_clr_WA

SSP245_CRE_CA = SSP245_CRE_sw_CA + SSP245_CRE_lw_CA
SSP245_CRE_SA = SSP245_CRE_sw_SA + SSP245_CRE_lw_SA
SSP245_CRE_WA = SSP245_CRE_sw_WA + SSP245_CRE_lw_WA

###################################################################
#######################################################################

Bias_net_CA = SAI_CRE_CA-SSP245_CRE_CA
Bias_net_WA = SAI_CRE_WA-SSP245_CRE_WA
Bias_net_SA = SAI_CRE_SA-SSP245_CRE_SA

Bias_net_sw_CA = SAI_CRE_sw_CA - SSP245_CRE_sw_CA
Bias_net_sw_WA = SAI_CRE_sw_WA - SSP245_CRE_sw_WA
Bias_net_sw_SA = SAI_CRE_sw_SA - SSP245_CRE_sw_SA

Bias_net_lw_CA = SAI_CRE_lw_CA - SSP245_CRE_lw_CA
Bias_net_lw_WA = SAI_CRE_lw_WA - SSP245_CRE_lw_WA
Bias_net_lw_SA = SAI_CRE_lw_SA - SSP245_CRE_lw_SA


################################ total clouds

fig,ax = plt.subplots(3,3, figsize=(15,10), tight_layout = True)
scale = 100


ax[0,0].scatter(Bias_Tot_WA, Bias_net_sw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_WA, Bias_net_sw_WA)
ax[0,0].plot(Bias_Tot_WA, intercept + slope*Bias_Tot_WA, 'r')
# a, b = np.polyfit(Bias_Tot_WA, Bias_net_sw_WA, deg=1)
# y_est = a*Bias_Tot_WA+b
# ax[0,0].plot(Bias_Tot_WA, y_est, 'k-')
#pearson_coef, p_value1 = stats.pearsonr(Bias_Tot_WA, Bias_net_sw_WA)
ax[0,0].text(-15, -5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,0].text(-15, -7.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')
ax[0,0].axvline(0, color= 'k', linestyle ='--')
ax[0,0].axhline(0, color= 'k', linestyle ='--')

ax[0,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,0].set_title('a)SWA         SW', fontsize=20, fontweight = 'bold')


ax[0,1].scatter(Bias_Tot_WA, Bias_net_lw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_WA, Bias_net_lw_WA)
ax[0,1].plot(Bias_Tot_WA, intercept + slope*Bias_Tot_WA, 'r')
ax[0,1].axvline(0, color= 'k', linestyle ='--')
ax[0,1].axhline(0, color= 'k', linestyle ='--')
ax[0,1].text(-15, 2.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,1].text(-15, 2, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[0,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,1].set_title('b) SWA         LW', fontsize=20, fontweight = 'bold')


ax[0,2].scatter(Bias_Tot_WA, Bias_net_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_WA, Bias_net_WA)
ax[0,2].plot(Bias_Tot_WA, intercept + slope*Bias_Tot_WA, 'r')
ax[0,2].axvline(0, color= 'k', linestyle ='--')
ax[0,2].axhline(0, color= 'k', linestyle ='--')
ax[0,2].text(-15, -5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,2].text(-15, -7.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')
ax[0,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,2].set_title('c) SWA         NET', fontsize=20, fontweight = 'bold')

################### Sahara region
ax[1,0].scatter(Bias_Tot_SA, Bias_net_sw_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_SA, Bias_net_sw_SA)
ax[1,0].plot(Bias_Tot_SA, intercept + slope*Bias_Tot_SA, 'r')
ax[1,0].axvline(0, color= 'k', linestyle ='--')
ax[1,0].axhline(0, color= 'k', linestyle ='--')
ax[1,0].text(10, 4, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[1,0].text(10, 3, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[1,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,0].set_title('d) SAH         SW', fontsize=20, fontweight = 'bold')

ax[1,1].scatter(Bias_Tot_SA, Bias_net_lw_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_SA, Bias_net_lw_SA)
ax[1,1].plot(Bias_Tot_SA, intercept + slope*Bias_Tot_SA, 'r')
ax[1,1].axvline(0, color= 'k', linestyle ='--')
ax[1,1].axhline(0, color= 'k', linestyle ='--')
ax[1,1].text(-12, 2.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[1,1].text(-12, 1.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[1,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,1].set_title('e) SAH         LW', fontsize=20, fontweight = 'bold')


ax[1,2].scatter(Bias_Tot_SA, Bias_net_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_SA, Bias_net_SA)
ax[1,2].plot(Bias_Tot_SA, intercept + slope*Bias_Tot_SA, 'r')
ax[1,2].axvline(0, color= 'k', linestyle ='--')
ax[1,2].axhline(0, color= 'k', linestyle ='--')
ax[1,2].text(8, 3.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[1,2].text(8, 2.5, 'P <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[1,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,2].set_title('f) SAH         NET', fontsize=20, fontweight = 'bold')

####################### Central Africa
ax[2,0].scatter(Bias_Tot_CA, Bias_net_sw_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_CA, Bias_net_sw_CA)
ax[2,0].plot(Bias_Tot_CA, intercept + slope*Bias_Tot_CA, 'r')
ax[2,0].axvline(0, color= 'k', linestyle ='--')
ax[2,0].axhline(0, color= 'k', linestyle ='--')
ax[2,0].text(-9, -8, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,0].text(-9, -12, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,0].set_title('g) CA         SW', fontsize=20, fontweight = 'bold')


ax[2,1].scatter(Bias_Tot_CA, Bias_net_lw_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_CA, Bias_net_lw_CA)
ax[2,1].plot(Bias_Tot_CA, intercept + slope*Bias_Tot_CA, 'r')
ax[2,1].axvline(0, color= 'k', linestyle ='--')
ax[2,1].axhline(0, color= 'k', linestyle ='--')
ax[2,1].text(-9, 3.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,1].text(-9, 3, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,1].set_title('h) CA         LW', fontsize=20, fontweight = 'bold')

ax[2,2].scatter(Bias_Tot_CA, Bias_net_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_Tot_CA, Bias_net_CA)
ax[2,2].plot(Bias_Tot_CA, intercept + slope*Bias_Tot_CA, 'r')
ax[2,2].axvline(0, color= 'k', linestyle ='--')
ax[2,2].axhline(0, color= 'k', linestyle ='--')
ax[2,2].text(-9, -3, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,2].text(-9, -6, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,2].set_title('i) CA         NET', fontsize=20, fontweight = 'bold')

ax[2,0].set_xlabel('ΔTCF (%)', fontsize=20, fontweight = 'bold')
ax[2,1].set_xlabel('ΔTCF (%)', fontsize=20, fontweight = 'bold')
ax[2,2].set_xlabel('ΔTCF (%)', fontsize=20, fontweight = 'bold')

plt.savefig("D:/SAI_Data/Figure_5.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_5.pdf", dpi=700, bbox_inches='tight')


###############################################################################
###############################################################################
##################################### low cloud 
###############################################################################
###############################################################################

fig,ax = plt.subplots(3,3, figsize=(15,10), tight_layout = True)
scale = 100

#AX = ax.twinx() #.twiny()

ax[0,0].scatter(Bias_low_WA, Bias_net_sw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_low_WA, Bias_net_sw_WA)
ax[0,0].plot(Bias_low_WA, intercept + slope*Bias_low_WA, 'r')
ax[0,0].axvline(0, color= 'k', linestyle ='--')
ax[0,0].axhline(0, color= 'k', linestyle ='--')
# a, b = np.polyfit(Bias_low_WA, Bias_net_sw_WA, deg=1)
# y_est = a*Bias_low_WA+b
# ax[0,0].plot(Bias_low_WA, y_est, 'k-')
pearson_coef, p_value1 = stats.pearsonr(Bias_low_WA, Bias_net_sw_WA)
ax[0,0].text(-4.5, -3, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,0].text(-4.5, -5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')
ax[0,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,0].set_title('a) SWA         SW', fontsize=20, fontweight = 'bold')


ax[0,1].scatter(Bias_low_WA, Bias_net_lw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_low_WA, Bias_net_lw_WA)
ax[0,1].plot(Bias_low_WA, intercept + slope*Bias_low_WA, 'r')
ax[0,1].axvline(0, color= 'k', linestyle ='--')
ax[0,1].axhline(0, color= 'k', linestyle ='--')
ax[0,1].text(-4.5, 2.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,1].text(-4.5, 2, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')
ax[0,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,1].set_title('b) SWA         LW', fontsize=20, fontweight = 'bold')


ax[0,2].scatter(Bias_low_WA, Bias_net_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_low_WA, Bias_net_WA)
ax[0,2].plot(Bias_low_WA, intercept + slope*Bias_low_WA, 'r')
ax[0,2].axvline(0, color= 'k', linestyle ='--')
ax[0,2].axhline(0, color= 'k', linestyle ='--')
ax[0,2].text(-4.5, -1.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[0,2].text(-4.5, -3.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[0,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,2].set_title('c) SWA         NET', fontsize=20, fontweight = 'bold')

################### Sahara region
ax[1,0].scatter(Bias_low_SA, Bias_net_sw_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_low_SA, Bias_net_sw_SA)
ax[1,0].plot(Bias_low_SA, intercept + slope*Bias_low_SA, 'r')
ax[1,0].axvline(0, color= 'k', linestyle ='--')
ax[1,0].axhline(0, color= 'k', linestyle ='--')
ax[1,0].text(0.25, 4, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15,fontweight = 'bold')
ax[1,0].text(0.25, 3, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15,fontweight = 'bold')
ax[1,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,0].set_title('d) SAH         SW', fontsize=20, fontweight = 'bold')

ax[1,1].scatter(Bias_low_SA, Bias_net_lw_SA,c='k',s=scale,alpha=1,marker='o')
ax[1,1].axvline(0, color= 'k', linestyle ='--')
ax[1,1].axhline(0, color= 'k', linestyle ='--')
slope, intercept, r, p, se =  stats.linregress(Bias_low_SA, Bias_net_lw_SA)
ax[1,1].plot(Bias_low_SA, intercept + slope*Bias_low_SA, 'r')
ax[1,1].text(-0.8, 3, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[1,1].text(-0.8, 2.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[1,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,1].set_title('e) SAH         LW', fontsize=20, fontweight = 'bold')


ax[1,2].scatter(Bias_low_SA, Bias_net_SA,c='k',s=scale,alpha=1,marker='o')
ax[1,2].axvline(0, color= 'k', linestyle ='--')
ax[1,2].axhline(0, color= 'k', linestyle ='--')
slope, intercept, r, p, se =  stats.linregress(Bias_low_SA, Bias_net_SA)
ax[1,2].plot(Bias_low_SA, intercept + slope*Bias_low_SA, 'r')
ax[1,2].text(-0.8, -1, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[1,2].text(-0.8, -2, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[1,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,2].set_title('f) SAH         NET', fontsize=20, fontweight = 'bold')

####################### Central Africa
ax[2,0].scatter(Bias_low_CA, Bias_net_sw_CA,c='k',s=scale,alpha=1,marker='o')
ax[2,0].axvline(0, color= 'k', linestyle ='--')
ax[2,0].axhline(0, color= 'k', linestyle ='--')
slope, intercept, r, p, se =  stats.linregress(Bias_low_CA, Bias_net_sw_CA)
ax[2,0].plot(Bias_low_CA, intercept + slope*Bias_low_CA, 'r')
ax[2,0].text(-3.5, -5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,0].text(-3.5, -7.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,0].set_title('g) CA         SW', fontsize=20, fontweight = 'bold')


ax[2,1].scatter(Bias_low_CA, Bias_net_lw_CA,c='k',s=scale,alpha=1,marker='o')
ax[2,1].axvline(0, color= 'k', linestyle ='--')
ax[2,1].axhline(0, color= 'k', linestyle ='--')
slope, intercept, r, p, se =  stats.linregress(Bias_low_CA, Bias_net_lw_CA)
ax[2,1].plot(Bias_low_CA, intercept + slope*Bias_low_CA, 'r')
ax[2,1].text(-3.5, 3, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,1].text(-3.5, 2.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,1].set_title('h) CA         LW', fontsize=20, fontweight = 'bold')

ax[2,2].scatter(Bias_low_CA, Bias_net_CA,c='k',s=scale,alpha=1,marker='o')
ax[2,2].axvline(0, color= 'k', linestyle ='--')
ax[2,2].axhline(0, color= 'k', linestyle ='--')
slope, intercept, r, p, se =  stats.linregress(Bias_low_CA, Bias_net_CA)
ax[2,2].plot(Bias_low_CA, intercept + slope*Bias_low_CA, 'r')
ax[2,2].text(-3.5, -4, 'r =' ' ' + '{:.2f}'.format(r), fontsize=15, fontweight = 'bold')
ax[2,2].text(-3.5, -6, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=15, fontweight = 'bold')

ax[2,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,2].set_title('i) CA         NET', fontsize=20, fontweight = 'bold')

ax[2,0].set_xlabel('ΔLCF (%)', fontsize=20, fontweight = 'bold')
ax[2,1].set_xlabel('ΔLCF (%)', fontsize=20, fontweight = 'bold')
ax[2,2].set_xlabel('ΔLCF (%)', fontsize=20, fontweight = 'bold')

plt.savefig("D:/SAI_Data/Figure_6.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_6.pdf", dpi=700, bbox_inches='tight')





