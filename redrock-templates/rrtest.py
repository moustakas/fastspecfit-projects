#!/usr/bin/env python

"""
rrdesi -i /global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/redux-templates-NMF-0.1-zscan01/vitiles/coadd-6-80613.fits \
  --targetids 39633341686743610 -o zbest.fits -d zdetails.h5 \
  -t /global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/rrtemplates/NMF-0.1

rrdesi -i /global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/redux-templates-NMF-0.1-zscan01/vitiles/coadd-6-80613.fits   \
  --targetids 39633341686743610 -o zbest.fits -d zdetails.h5   -t temp --zscan-galaxy="0.3,0.4,3e-4"

from redrock.results import read_zscan
zs, zf = read_zscan('zdetails.h5') # sqrt(ivar) weights
zs2, zf2 = read_zscan('zdetails2.h5') # ivar weights

"""
import os, pdb
import fitsio
from astropy.table import Table
import numpy as np
import redrock.templates
from redrock.rebin import trapz_rebin
from desispec.io.spectra import read_spectra
from desispec.resolution import Resolution
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter
    
ss = read_spectra('/global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/redux-templates-NMF-0.1-zscan01/vitiles/coadd-6-80613.fits',
                  targetids=39633341686743610)

cams = ['b', 'r', 'z']
wave = [ss.wave[cam] for cam in cams]
flux = [ss.flux[cam].squeeze() for cam in cams]
ivar = [ss.ivar[cam].squeeze() for cam in cams]
R = [Resolution(ss.resolution_data[cam][0, :, :].squeeze()) for cam in cams]
                        
# interpolate the templates onto the data at the "correct" redshift
zstar = 3.52640274513618e-5
zgal = 0.3330972364229895
#S = redrock.templates.Template('/global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/rrtemplates/NMF-0.1/rrtemplate-star-K.fits')
G = redrock.templates.Template('/global/cfs/cdirs/desi/users/ioannis/fastspecfit/redrock-templates/stacks/rrtemplates/NMF-0.1/rrtemplate-galaxy-NMF-0.1.fits')

Gmodelflux = []
for icam in range(len(cams)):
    Gmodelflux.append(R[icam].dot(trapz_rebin(G.wave*(1.+zgal), G.flux/(1.+zgal), wave[icam])))
vGmodelflux = np.vstack(Gmodelflux)    

if False:
    ii = 6
    plt.clf()
    plt.plot(G.wave*(1.+zgal), G.flux[ii, :]/(1.+zgal), color='k')
    for icam in range(len(cams)):
        plt.plot(wave[icam], Gmodelflux[icam][:, ii], color='red')
    #plt.xlim(3600, 9800)
    plt.xlim(6500, 6900)
    plt.ylim(-0.05, 0.2)
    plt.savefig('ioannis/tmp/junk.png')

maxiter = 100000
atol = 1e-8

# vGmodelflux [npix,ntemplate] = Tb [npix,ntemplate]
# Redrock implementation of nnls (see note by Dylan Green)
weights = np.hstack(ivar)
wflux = weights * np.hstack(flux)
M = vGmodelflux.T.dot(np.multiply(weights[:, None], vGmodelflux))
y = vGmodelflux.T.dot(wflux)
coeff_rr, rnorm_rr = nnls(M, y, maxiter=maxiter)#, atol=atol)
chi2_rr = np.sum(np.hstack(ivar) * (np.hstack(flux) - vGmodelflux.dot(coeff_rr))**2)
print('coeff (redrock)', coeff_rr)
print('chi2, rnorm (redrock)', chi2_rr, rnorm_rr)

inverr = np.sqrt(np.hstack(ivar))
inverr = np.hstack(ivar)
bvector = np.hstack(flux) * inverr
Amatrix = vGmodelflux * inverr[:, np.newaxis]
coeff, rnorm = nnls(A=Amatrix, b=bvector, maxiter=maxiter)#, atol=atol)
chi2 = np.sum(np.hstack(ivar) * (np.hstack(flux) - vGmodelflux.dot(coeff))**2)
print('coeff (my nnls)', coeff)
print('chi2, rnorm (my nnls)', chi2, rnorm)

if True:
    plt.clf()
    plt.plot(np.hstack(wave), gaussian_filter(np.hstack(flux), 2), color='gray')
    plt.plot(np.hstack(wave), gaussian_filter(vGmodelflux.dot(coeff), 2), color='blue', alpha=1,
             label=r'$\chi^{2}=||V^{1/2}Wh-V^{1/2}x||^{2}$='+f'{chi2:.2f}')
    plt.plot(np.hstack(wave), gaussian_filter(vGmodelflux.dot(coeff_rr), 2), color='red', alpha=0.3,
             label=r'$\chi^{2}=||W^{T}VWh-W^{T}Vx||^{2}$='+f'{chi2_rr:.2f}')
    plt.xlim(3600, 9800)
    plt.xlabel('Wavelength ($\AA$)')
    plt.ylabel(r'$F_{\lambda}$')
    #plt.xlim(6500, 6900)
    plt.ylim(-2, 10)
    plt.legend()
    plt.savefig('ioannis/tmp/junk.png')

pdb.set_trace()



