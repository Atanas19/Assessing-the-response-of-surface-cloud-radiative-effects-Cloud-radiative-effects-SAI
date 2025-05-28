# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:58:00 2024

@author: ad7gb
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

########################################################################
##############################################################################
#############################################################################


f = 'D:/SAI_Data/SSP245/LWP/'
ddirsM = os.listdir(f)

#################################################################################
#################################################################################

LWP = xr.open_mfdataset(f+'EnsembleMean-b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.001-010.cam.h0.TGCLDLWP.201501-206912.nc')

LWP.coords['lon'] = (LWP.coords['lon'] + 180) % 360 - 180
LWP = LWP.sortby(LWP.lon)
laC  = LWP.lat.values
loC  = LWP.lon.values

################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
LWP_WA = LWP.sel(time = slice('2035-01-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
LWP_SA = LWP.sel(time = slice('2035-01-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
LWP_CA = LWP.sel(time = slice('2035-01-01','2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

###############################################################################
###############################################################################
########################### ARISE #############################################
###############################################################################
###############################################################################

f = 'D:/SAI_Data/ARISE_WACCM/LWP/'
ddirsM = os.listdir(f)

ALWP = xr.open_mfdataset(f+'EnsembleMean-b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001-010.cam.h0.TGCLDLWP.203501-206912.nc')

ALWP.coords['lon'] = (ALWP.coords['lon'] + 180) % 360 - 180
ALWP = ALWP.sortby(ALWP.lon)
laC  = ALWP.lat.values
loC  = ALWP.lon.values


################### red box
lat = laC[np.where((laC>=7)&(laC<=13))[0]]
lon = loC[np.where((loC>=-12)&(loC<=6))[0]]
ALWP_WA = ALWP.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
ALWP_SA = ALWP.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
ALWP_CA = ALWP.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['TGCLDLWP'].values*1000

########################################################################################################
########################################################################################################

Bias_LWP_CA = ALWP_CA - LWP_CA
Bias_LWP_WA = ALWP_WA - LWP_WA
Bias_LWP_SA = ALWP_SA - LWP_SA

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
SAI_lw_WA = SAI_lw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_SA = SAI_lw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_CA = SAI_lw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values


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
SAI_lw_clr_WA = SAI_lw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_lw_clr_SA = SAI_lw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_lw_clr_CA = SAI_lw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

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
SAI_sw_WA = SAI_sw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_SA = SAI_sw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_CA = SAI_sw_net.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values


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
SAI_sw_clr_WA = SAI_sw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SAI_sw_clr_SA = SAI_sw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SAI_sw_clr_CA = SAI_sw_net_clr.sel(time = slice('2035-01-01','2070-01-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values


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
SSP245_lw_WA = SSP245_lw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_SA = SSP245_lw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_CA = SSP245_lw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNS'].values


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
SSP245_lw_clr_WA = SSP245_lw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_lw_clr_SA = SSP245_lw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_lw_clr_CA = SSP245_lw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FLNSC'].values


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
SSP245_sw_WA = SSP245_sw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_SA = SSP245_sw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_CA = SSP245_sw_net.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNS'].values


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
SSP245_sw_clr_WA = SSP245_sw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### black box 
lat = laC[np.where((laC>=16)&(laC<=22))[0]]
lon = loC[np.where((loC>=-8)&(loC<=4))[0]]
SSP245_sw_clr_SA = SSP245_sw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values

#################### Green box 
lat = laC[np.where((laC>=-2)&(laC<=5))[0]]
lon = loC[np.where((loC>=10)&(loC<=26))[0]]
SSP245_sw_clr_CA = SSP245_sw_net_clr.sel(time = slice('2035-01-01', '2069-12-31'),lat = lat, lon = lon).mean(('lon','lat'))['FSNSC'].values


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

###############################################################################
###############################################################################

Bias_LWP_CA = ALWP_CA - LWP_CA
Bias_LWP_WA = ALWP_WA - LWP_WA
Bias_LWP_SA = ALWP_SA - LWP_SA



fig,ax = plt.subplots(3,3, figsize=(15,10), tight_layout = True)
scale = 100

#AX = ax.twinx() #.twiny()

ax[0,0].scatter(Bias_LWP_WA, Bias_net_sw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_WA, Bias_net_sw_WA)
ax[0,0].plot(Bias_LWP_WA, intercept + slope*Bias_LWP_WA, 'r')
ax[0,0].axvline(0, color= 'k', linestyle ='--')
ax[0,0].axhline(0, color= 'k', linestyle ='--')

# a, b = np.polyfit(Bias_LWP_WA, Bias_net_sw_WA, deg=1)
# y_est = a*Bias_LWP_WA+b
# ax[0,0].plot(Bias_LWP_WA, y_est, 'k-')

ax[0,0].text(0, 39, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[0,0].text(0, 31, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[0,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,0].set_title('a) SWA       SW', fontsize=20, fontweight = 'bold')


ax[0,1].scatter(Bias_LWP_WA, Bias_net_lw_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_WA, Bias_net_lw_WA)
ax[0,1].plot(Bias_LWP_WA, intercept + slope*Bias_LWP_WA, 'r')
ax[0,1].axvline(0, color= 'k', linestyle ='--')
ax[0,1].axhline(0, color= 'k', linestyle ='--')
ax[0,1].text(-60, 3.5, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[0,1].text(-60, 2, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[0,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,1].set_title('b) SWA       LW', fontsize=20, fontweight = 'bold')


ax[0,2].scatter(Bias_LWP_WA, Bias_net_WA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_WA, Bias_net_WA)
ax[0,2].plot(Bias_LWP_WA, intercept + slope*Bias_LWP_WA, 'r')
ax[0,2].axvline(0, color= 'k', linestyle ='--')
ax[0,2].axhline(0, color= 'k', linestyle ='--')
ax[0,2].text(0, 31, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[0,2].text(0, 24, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')

ax[0,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[0,2].set_title('c) SWA       NET', fontsize=20, fontweight = 'bold')

################### Sahara region
ax[1,0].scatter(Bias_LWP_SA, Bias_net_sw_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_SA, Bias_net_sw_SA)
ax[1,0].plot(Bias_LWP_SA, intercept + slope*Bias_LWP_SA, 'r')
ax[1,0].axvline(0, color= 'k', linestyle ='--')
ax[1,0].axhline(0, color= 'k', linestyle ='--')
ax[1,0].text(-20, -3, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[1,0].text(-20, -8, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[1,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,0].set_title('d) SAH       SW', fontsize=20, fontweight = 'bold')

ax[1,1].scatter(Bias_LWP_SA, Bias_net_lw_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_SA, Bias_net_lw_SA)
ax[1,1].plot(Bias_LWP_SA, intercept + slope*Bias_LWP_SA, 'r')
ax[1,1].axvline(0, color= 'k', linestyle ='--')
ax[1,1].axhline(0, color= 'k', linestyle ='--')
ax[1,1].text(-20, 4, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[1,1].text(-20, 2.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[1,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,1].set_title('e) SAH       LW', fontsize=20, fontweight = 'bold')


ax[1,2].scatter(Bias_LWP_SA, Bias_net_SA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_SA, Bias_net_SA)
ax[1,2].plot(Bias_LWP_SA, intercept + slope*Bias_LWP_SA, 'r')
ax[1,2].axvline(0, color= 'k', linestyle ='--')
ax[1,2].axhline(0, color= 'k', linestyle ='--')
ax[1,2].text(-20, -1, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[1,2].text(-20, -3, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')

ax[1,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[1,2].set_title('f) SAH       NET', fontsize=20, fontweight = 'bold')

####################### Central Africa
ax[2,0].scatter(Bias_LWP_CA, Bias_net_sw_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_CA, Bias_net_sw_CA)
ax[2,0].plot(Bias_LWP_CA, intercept + slope*Bias_LWP_CA, 'r')
ax[2,0].axvline(0, color= 'k', linestyle ='--')
ax[2,0].axhline(0, color= 'k', linestyle ='--')
ax[2,0].text(0, 35, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[2,0].text(0, 28, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[2,0].set_ylabel('ΔSWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,0].set_title('g) CA       SW', fontsize=20, fontweight = 'bold')


ax[2,1].scatter(Bias_LWP_CA, Bias_net_lw_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_CA, Bias_net_lw_CA)
ax[2,1].plot(Bias_LWP_CA, intercept + slope*Bias_LWP_CA, 'r')
ax[2,1].axvline(0, color= 'k', linestyle ='--')
ax[2,1].axhline(0, color= 'k', linestyle ='--')
ax[2,1].text(-40, 4, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[2,1].text(-40, 2.5, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[2,1].set_ylabel('ΔLWCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,1].set_title('h) CA       LW', fontsize=20, fontweight = 'bold')

ax[2,2].scatter(Bias_LWP_CA, Bias_net_CA,c='k',s=scale,alpha=1,marker='o')
slope, intercept, r, p, se =  stats.linregress(Bias_LWP_CA, Bias_net_CA)
ax[2,2].plot(Bias_LWP_CA, intercept + slope*Bias_LWP_CA, 'r')
ax[2,2].axvline(0, color= 'k', linestyle ='--')
ax[2,2].axhline(0, color= 'k', linestyle ='--')
ax[2,2].text(0, 30, 'r =' ' ' + '{:.2f}'.format(r), fontsize=20, fontweight = 'bold')
ax[2,2].text(0, 20, 'p <' ' ' + '{:.3f}'.format(0.001), fontsize=20, fontweight = 'bold')
ax[2,2].set_ylabel('ΔNETCRE (W/m²)', fontsize=20, fontweight = 'bold')
ax[2,2].set_title('i) CA       NET', fontsize=20, fontweight = 'bold')

ax[2,0].set_xlabel('ΔLWP (g/m²)', fontsize=20, fontweight = 'bold')
ax[2,1].set_xlabel('ΔLWP (g/m²)', fontsize=20, fontweight = 'bold')
ax[2,2].set_xlabel('ΔLWP (g/m²)', fontsize=20, fontweight = 'bold')

plt.savefig("D:/SAI_Data/Figure_4.png", dpi=700, bbox_inches='tight')
plt.savefig("D:/SAI_Data/Figure_4.pdf", dpi=700, bbox_inches='tight')
