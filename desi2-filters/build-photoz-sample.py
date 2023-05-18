#!/usr/bin/env python

"""Synthesize photometry in five medium-band filters (centered around g-band)
using the FastSpecFit/Iron SV3 models.

source /global/cfs/cdirs/desi/software/desi_environment.sh 23.1
module swap desispec/0.57.0
module load fastspecfit/2.1.1

cd /global/cfs/cdirs/desicollab/users/ioannis/desi2-filters
time python build-photoz-sample --mp 128

"""
import os, pdb
import numpy as np
import fitsio
import argparse    
from astropy.table import Table
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt

from speclite import filters
from fastspecfit.continuum import ContinuumTools
from fastspecfit.util import TabulatedDESI
from fastspecfit.io import read_fastspecfit, write_fastspecfit, cache_templates, FLUXNORM

sns.set(context='talk', style='ticks', font_scale=1.1)#, rc=rc)

def _rebuild_one(args):
    return rebuild_one(*args)

def rebuild_one(igal, redshift, dlum, coeff, filt):
    print('Rebuilding {}'.format(igal))
    templatecache = cache_templates()

    CTools = ContinuumTools(continuum_pixkms=templatecache['continuum_pixkms'],
                            pixkms_wavesplit=templatecache['pixkms_wavesplit'])

    sedwave = templatecache['templatewave'] * (1 + redshift)
    T = CTools.transmission_Lyman(redshift, sedwave)
    T *= CTools.massnorm * (10.0 / (1e6 * dlum))**2 / (1.0 + redshift)

    sedmodel = coeff.dot(templatecache['templateflux'].T)
    sedmodel *= T

    synthmaggies = filt.get_ab_maggies(sedmodel, sedwave)
    synthmaggies = np.vstack(synthmaggies.as_array().tolist())[0]

    return synthmaggies
    
def add_bitnames(meta):

    desi_bitnames = np.zeros(len(meta), dtype='U300')
    bgs_bitnames = np.zeros(len(meta), dtype='U300')
    mws_bitnames = np.zeros(len(meta), dtype='U300')
    scnd_bitnames = np.zeros(len(meta), dtype='U300')
    cmx_bitnames = np.zeros(len(meta), dtype='U300')
    targetclass = np.zeros(len(meta), dtype='U300')

    def get_targetclass(targetclass, name):
        for cc in ['BGS', 'LRG', 'ELG', 'QSO', 'MWS', 'SCND', 'STD']:
            if cc in name:
                for iobj, tclass in enumerate(targetclass):
                    if tclass == '':
                        targetclass[iobj] = cc
                    else:
                        if not cc in tclass: # only once
                            targetclass[iobj] = ' '.join([tclass, cc])
        return targetclass
    
    for survey, prefix in zip(['SV3'], ['SV3_']):
    #for survey, prefix in zip(['CMX', 'SV1', 'SV2', 'SV3', 'MAIN'], ['CMX_', 'SV1_', 'SV2_', 'SV3_', '']):
        I = meta['SURVEY'] == survey.lower()
        if np.sum(I) > 0:
            if survey == 'MAIN':
                from desitarget.targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
            elif survey == 'SV1':
                from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
            elif survey == 'SV2':
                from desitarget.sv2.sv2_targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
            elif survey == 'SV3':
                from desitarget.sv3.sv3_targetmask import desi_mask, bgs_mask, mws_mask, scnd_mask
            elif survey == 'CMX':
                from desitarget.cmx.cmx_targetmask import cmx_mask

            if survey == 'CMX':
                for name in cmx_mask.names():
                    J = np.where(meta['CMX_TARGET'.format(prefix)] & cmx_mask.mask(name) != 0)[0]
                    if len(J) > 0:
                        cmx_bitnames[J] = [' '.join([bit, name]) for bit in cmx_bitnames[J]]
                        #if 'QSO' in name:
                        #    pdb.set_trace()
                        #print(name, targetclass[J])
                        targetclass[J] = get_targetclass(targetclass[J], name)
            else:
                for name in desi_mask.names():
                    J = np.where(meta['{}DESI_TARGET'.format(prefix)] & desi_mask.mask(name) != 0)[0]
                    if len(J) > 0:
                        desi_bitnames[J] = [' '.join([bit, name]) for bit in desi_bitnames[J]]
                        targetclass[J] = get_targetclass(targetclass[J], name)
                        
                for name in bgs_mask.names():
                    J = np.where(meta['{}BGS_TARGET'.format(prefix)] & bgs_mask.mask(name) != 0)[0]
                    if len(J) > 0:
                        bgs_bitnames[J] = [' '.join([bit, name]) for bit in bgs_bitnames[J]]
                        targetclass[J] = get_targetclass(targetclass[J], name)
                        
                for name in mws_mask.names():
                    J = np.where(meta['{}MWS_TARGET'.format(prefix)] & mws_mask.mask(name) != 0)[0]
                    if len(J) > 0:
                        mws_bitnames[J] = [' '.join([bit, name]) for bit in mws_bitnames[J]]
                        targetclass[J] = get_targetclass(targetclass[J], name)
                        
                for name in scnd_mask.names():
                    J = np.where(meta['{}SCND_TARGET'.format(prefix)] & scnd_mask.mask(name) != 0)[0]
                    if len(J) > 0:
                        scnd_bitnames[J] = [' '.join([bit, name]) for bit in scnd_bitnames[J]]
                        targetclass[J] = get_targetclass(targetclass[J], name)

    meta['DESI_BITNAMES'] = desi_bitnames
    meta['BGS_BITNAMES'] = bgs_bitnames
    meta['MWS_BITNAMES'] = mws_bitnames
    meta['SCND_BITNAMES'] = scnd_bitnames
    meta['CMX_BITNAMES'] = cmx_bitnames
    meta['TARGETCLASS'] = targetclass

def main():
    """Main wrapper."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mp', type=int, default=1, help='Number of multiprocessing processes per MPI rank or node.')
    args = parser.parse_args()

    specprod = 'fuji'
    version = 'v2.0'
    
    catdir = os.path.join(os.environ.get('DESI_ROOT'), 'spectro', 'fastspecfit', 
                          specprod, version, 'catalogs')
    
    fastfile = os.path.join(catdir, 'fastspec-{}.fits'.format(specprod))
    
    metacols = ['SURVEY', 'PROGRAM', 'ZWARN', 'COADD_FIBERSTATUS', 'Z',
                'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET', 
                'SV3_SCND_TARGET', 'FLUX_G', 'FLUX_R', 'FLUX_Z',
                'DELTACHI2']
    fastcols = ['RCHI2', 'APERCORR']
    
    meta = fitsio.read(fastfile, 'METADATA', columns=metacols)
    fast = fitsio.read(fastfile, 'FASTSPEC', columns=fastcols)

    # select a parent sample
    rows = np.where(np.logical_or(meta['PROGRAM'] == 'dark', meta['PROGRAM'] == 'bright') *
                    (meta['SURVEY'] == 'sv3') * (meta['ZWARN'] == 0) * 
                    (meta['DELTACHI2'] > 40) *
                    (meta['COADD_FIBERSTATUS'] == 0) * (meta['SV3_SCND_TARGET'] == 0) *
                    (meta['FLUX_G'] > 0) * (meta['FLUX_R'] > 0) * (meta['FLUX_Z'] > 0) * 
                    (fast['RCHI2'] < 10) * (fast['APERCORR'] > 1) * (fast['APERCORR'] < 3) * 
                    (meta['Z'] < 3))[0]
    rows = rows[np.argsort(rows)]
    print('Selecting a parent sample of {} objects'.format(len(rows)))
    
    meta = Table(fitsio.read(fastfile, 'METADATA', rows=rows))
    add_bitnames(meta)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    print('Parent selection:')
    for target in ['BGS', 'ELG', 'QSO', 'LRG']:
        #I = meta['TARGETCLASS'] == target
        I = np.isin(meta['TARGETCLASS'], target)
        print(target, np.sum(I))
        ax1.scatter(meta['Z'][I], 22.5-2.5*np.log10(meta['FLUX_R'][I]), 
                    s=10, marker='s', label=target, alpha=0.5)

    print('Refined selection:')
    S, targetclass = [], []
    for target in ['BGS', 'ELG', 'QSO', 'LRG']:
        if target == 'BGS':
            J = ((meta['TARGETCLASS'] == 'BGS') * (meta['FLUX_R'] > 10**(0.4*(22.5-20))) *
                 (meta['Z'] < 0.7))

        elif target == 'ELG':
            J = (np.isin(meta['TARGETCLASS'], 'ELG') * (meta['FLUX_R'] > 10**(0.4*(22.5-24))) *
                 (meta['Z'] > 0.6) * (meta['Z'] < 1.5))
        elif target == 'QSO':
            J = np.isin(meta['TARGETCLASS'], 'QSO') * (meta['Z'] > 1.4)
        elif target == 'LRG':
            J = ((meta['TARGETCLASS'] == 'LRG') * (meta['FLUX_R'] > 10**(0.4*(22.5-24))) *
                 (meta['Z'] > 0.2) * (meta['Z'] < 1.5))
        else:
            J = None

        if J is not None:
            print(target, np.sum(J))
            J = np.where(J)[0]
            S.append(J)
            targetclass.append([target] * len(J))
            ax2.scatter(meta['Z'][J], 22.5-2.5*np.log10(meta['FLUX_R'][J]), 
                        s=10, marker='s', label=target, alpha=0.5)

    # select a subset of the refined sample
    S = np.hstack(S)
    targetclass = np.hstack(targetclass)
    _, uindx = np.unique(S, return_index=True)
    S = S[uindx]
    targetclass = targetclass[uindx]
    print('Refined sample: N={}'.format(len(S)))

    #rows = rows[S]
    rand = np.random.RandomState(seed=1)
    R = [] # final index
    for target in ['BGS', 'ELG', 'QSO', 'LRG']:
        T = np.where(targetclass == target)[0]
        if target == 'QSO':
            size = len(T)
        else:
            size = int(0.1*len(T))

        if size > 1000:
            size = 1000
        C = rand.choice(len(T), size=size, replace=False)
        print(target, len(C))
        R.append(T[C])

        ax3.scatter(meta['Z'][S][T][C], 22.5-2.5*np.log10(meta['FLUX_R'][S][T][C]), 
                    s=20, marker='s', label=target, alpha=0.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(14, 25)
        ax.set_ylabel('r (AB mag)')
    ax3.legend(markerscale=2, prop={'size': 12})
    ax3.set_xlabel('Redshift')

    fig.subplots_adjust(bottom=0.1, right=0.95, left=0.15, top=0.95, hspace=0.07)
    fig.savefig('fastspec-desi2.png')

    # write out the sample
    R = np.hstack(R)

    if False:
        finalfast, finalmeta, coadd_type, _ = read_fastspecfit(fastfile, rows=rows[S][R])
    else:
        #print('Hack!')
        #S = S[:256]
        finalfast, finalmeta, coadd_type, _ = read_fastspecfit(fastfile, rows=rows[S])
    ngal = len(finalfast)

    outfile = 'fastspec-desi2.fits.gz'
    if not os.path.isfile(outfile):
        write_fastspecfit(finalfast, finalmeta, outfile=outfile, 
                          specprod=specprod, coadd_type=coadd_type)
    
    #I = np.sort(S[R])
    #assert(np.all(meta2['TARGETID'] == meta[I]['TARGETID']))
    #plt.clf() ; plt.scatter(meta['Z'][S][R], 22.5-2.5*np.log10(meta['FLUX_R'][S][R]), s=5) ; plt.savefig('junk.png')

    # synthesize photometry on a redshift grid
    cosmo = TabulatedDESI()

    filt = filters.FilterSequence((
        filters.load_filter('desi2-g1.ecsv'), filters.load_filter('desi2-g2.ecsv'),
        filters.load_filter('desi2-g3.ecsv'), filters.load_filter('desi2-g4.ecsv'),
        filters.load_filter('desi2-g5.ecsv'), filters.load_filter('decam2014-g'),
        filters.load_filter('decam2014-r'), filters.load_filter('decam2014-z'), 
        filters.load_filter('wise2010-W1'), filters.load_filter('wise2010-W2'),
        filters.load_filter('wise2010-W3'), filters.load_filter('wise2010-W4')
    ))
    nfilt = len(filt)

    if False:
        templatecache = cache_templates()

        CTools = ContinuumTools(continuum_pixkms=templatecache['continuum_pixkms'],
                                pixkms_wavesplit=templatecache['pixkms_wavesplit'])

        zgrid = np.linspace(1e-3, 3, 5)
        dlum = cosmo.luminosity_distance(zgrid)
    
        phot = []
        for iz, redshift in enumerate(zgrid):
            print('Working on redshift {:.3f}'.format(redshift))
            sedwave = templatecache['templatewave'] * (1 + redshift)
            T = CTools.transmission_Lyman(redshift, sedwave)
            T *= CTools.massnorm * (10.0 / (1e6 * dlum[iz]))**2 / (1.0 + redshift)
    
            sedmodels = finalfast['COEFF'].data.dot(templatecache['templateflux'].T) # [ngal, nwave]
            sedmodels *= T[np.newaxis, :]
    
            synthmaggies = filt.get_ab_maggies(sedmodels, sedwave)
            synthmaggies = np.vstack(synthmaggies.as_array().tolist()).T
            phot.append(synthmaggies)
    
            if False:
                synthphot = CTools.parse_photometry(filt.names, synthmaggies, 
                                                    filt.effective_wavelengths,
                                                    nanomaggies=False)
    
        phot = np.stack(phot, axis=1)
        hdr = fitsio.FITSHDR()
        hdr['ZMIN'] = zgrid[0]
        hdr['ZMAX'] = zgrid[-1]
        hdr['NZ'] = len(zgrid)
        hdr['NFILT'] = len(filt)
        for ifilt, name in enumerate(filt.names):
            hdr['FILT{}'.format(ifilt+1)] = name
    
        outfile = 'desi2-synthphot.fits'
        print('Writing {}'.format(outfile))
        fitsio.write(outfile, phot.astype('f4'), header=hdr, clobber=True)
    else:
        redshifts = finalmeta['Z'].data
        coeffs = finalfast['COEFF'].data
        dlums = cosmo.luminosity_distance(redshifts)

        mpargs = []
        for igal, (redshift, dlum) in enumerate(zip(redshifts, dlums)):
            mpargs.append([igal, redshift, dlum, coeffs[igal, :], filt])
                  
        if args.mp > 1:
            with multiprocessing.Pool(args.mp) as P:
                phot = P.map(_rebuild_one, mpargs)
        else:
            phot = [rebuild_one(*mparg) for mparg in mpargs]

        phot = np.stack(phot).T
        
        out = np.zeros((len(filt)+1, len(redshifts)), dtype='f4')
        out[0:, :] = redshifts
        out[1:, :] = phot
        del phot

        outfile = 'desi2-synthphot.fits'
        print('Writing {}'.format(outfile))
        fitsio.write(outfile, out, clobber=True)

if __name__ == '__main__':
    main()

