# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Python code to perform PCA and bayesian regresion analysis on M dwarfs
# jmaldonado at inaf-oapa, last change: September 16, 2020
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from PyAstronomy import pyasl
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import pandas
import numpy as np 
import statsmodels.api as sm
from scipy.stats import uniform, norm
import pylab as plt

import os
import statistics
from statistics import stdev 
from scipy.stats import uniform, norm
import scipy.optimize as op
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Help text
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Python code to perform PCA and bayesian regresion analysis on M dwarfs   ')
print('jmaldonado at inaf-oapa, last change: September 16, 2020                 ') 
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Definition of global variables
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
spectra_dir           = os.getcwd()+'/fits/'
ifile                 = 'MDWARFS_metal_ifile_for_pca_ver1.0.csv' 
mdwarfs_data          = 'MDWARFS_metal_calibration_data_all_elements_ver1.0.csv'
ofile                 = 'MDWARFS_abundances_with_indexes_pca_all_elements_example_output.txt'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read the abundances of the calibration stars
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
calibration_data=pandas.read_csv(mdwarfs_data)
n_training = 16
n_ions     = 15
max_n_pca  = n_training - 2 
stellar_abundance = calibration_data.iloc[0:n_training,0:n_ions].values.astype(float)

code=['[Fe/H]','[C/H]', '[Na/H]','[Mg/H]', '[Al/H]', '[Si/H]','[Ca/H]','[ScII/H]','[TiI/H]','[V/H]','[CrI/H]','[Mn/H]','[Co/H]','[Ni/H]', '[Zn/H]']
ion =[26.0, 6.0, 8.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.1, 22.0, 23.0, 24.0, 25.0, 27.0, 28.0, 30.0]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Reading input spectra 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fits_name = pandas.read_table(ifile) 
ifits = fits_name[fits_name.columns[0]]
n_stars = len(ifits)

# Spectra should be rebinned to a common wavelength scale
grid_step  = 0.01
grid_start = 5340. 
grid_end   = 6900.
grid_wavelengths = np.arange(grid_start,grid_end,grid_step) 
grid_nwavelengths = len(grid_wavelengths)
stellar_rebin_flux = np.zeros((grid_nwavelengths, n_stars))

for ii in range(0, n_stars):
   fits_now = spectra_dir + ifits[ii]
   wvl, flx = pyasl.read1dFitsSpec(fits_now)
   inter_flx = griddata(wvl, flx, grid_wavelengths, method='cubic')
   stellar_rebin_flux[:,ii] = inter_flx 

# Smooth the spectra to a lower resolution 
# [A gaussian smooth with sigma = 120 is performed] 
smoothed_flux_matrix = np.zeros((grid_nwavelengths, n_stars))
for ii in range(0, n_stars):
   gaussian_filter(stellar_rebin_flux[:,ii],120, output=smoothed_flux_matrix[:,ii])

# Get all the spectra to the same flux level using a  pseudo continuum centred in the R band
stellar_flux_matrix  = np.zeros((grid_nwavelengths, n_stars))
smoothed_refe_flux = smoothed_flux_matrix[:,0]
R_band_center = 6090
R_band_width  = 20
result = np.where(np.logical_and(grid_wavelengths> R_band_center - (R_band_width/2.0), grid_wavelengths< R_band_center + (R_band_width/2.0)))
R_ref = smoothed_refe_flux[result]

for ii in range(0, n_stars):
   smoothed_flux_now = smoothed_flux_matrix[:,ii]
   R_flux = smoothed_flux_now[result].reshape((-1, 1))
   model = LinearRegression().fit(R_flux,R_ref)
   stellar_flux_matrix[:,ii] = smoothed_flux_matrix[:,ii]*model.coef_ + model.intercept_

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# PCA analysis to determine [Fe/H]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
pca_fitted_metal  = np.zeros((n_stars,n_ions))
pca_fitted_emetal = np.zeros((n_stars,n_ions))

f = open(ofile, "w+")
print('# Results on PCA analysis with DRA analysis of M dwarf abundances',file=f)
import datetime
from datetime import date
datetime_object = datetime.datetime.now()
print("#", datetime_object, file=f)
print('# ++++++++++++++++++++++++++++++++++++++',file=f) 
print('# TRAINING DATASET list ----------------',file=f)
print(ifits[0:n_training],file=f)
print('# ++++++++++++++++++++++++++++++++++++++',file=f)
np.set_printoptions(precision=4)

# Perform the PCA analysis
stellar_flux_matrix = StandardScaler().fit_transform(stellar_flux_matrix)    
pca = PCA(n_components=max_n_pca)
principalComponents = pca.fit_transform(stellar_flux_matrix)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fit using ARD regression 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

for jj in range(0,n_ions):

    # Y_training 
    abundance_now = stellar_abundance[:,jj]
    y_training    = abundance_now.ravel(order='C') 

    y_training = y_training[0:n_training]
    # not all abundances are available for all stars 
    index = np.argwhere(np.isfinite(y_training))
    y_training = y_training[index]
    n_valid_abd  = len(y_training)
    y_training = y_training.ravel(order='C')

    # number of PCAs to be used, at maximum n_good_training - 2 
    n_components = n_valid_abd - 2
    X = pca.components_[0:n_components,:]
    X = np.transpose(X)

    # selection of X training from X
    X_training = pca.components_[0:n_components,0:n_training]  
    X_training = X_training[:,index]
    X_training = np.reshape(X_training, (n_components, n_valid_abd))
    X_training = np.transpose(X_training)

    # real, regression comes here
    clf = ARDRegression(compute_score=True)
    clf.fit(X_training,y_training)

    pca_fitted_metal[:,jj]  = clf.predict(X)

print('# ++++++++++++++++++++++++++++++++++++++',file=f)
print('# Derived abundances for the TRAINING stars',file=f)
print('#', code[:], file=f)
for ii in range(0, n_training):
    y = []
    arr = np.array(pca_fitted_metal[ii,:])
    for jj in arr: 
        x = '{:9.2f}'.format(jj)
        y.append(x)
    print (y,file=f)


print('# ++++++++++++++++++++++++++++++++++++++',file=f)
print('# Derived abundances for the PROBLEM stars',file=f)
print('#', code[:], file=f)
for ii in range(n_training, n_stars):
     y = []
     arr = np.array(pca_fitted_metal[ii,:])
     for jj in arr:
         x = '{:9.2f}'.format(jj)
         y.append(x)
     print (y,file=f)


f.close()

