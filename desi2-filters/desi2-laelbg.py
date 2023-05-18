#!/usr/bin/env python

import os, pdb
import fitsio
import numpy as np
import fsps
from argparse import ArgumentParser

from astropy.table import Table, vstack
from astropy import constants, units
from astropy.io import fits
from scipy.special import erf

from speclite import filters
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='talk', style='ticks', font_scale=0.7, palette='Set1')
colors = sns.color_palette()

from fastspecfit.util import TabulatedDESI

def igm_transmission(wavelength, redshift):
    """Intergalactic transmission (Meiksin, 2006)

    Compute the intergalactic transmission as described in Meiksin, 2006.

    Parameters
    ----------
    wavelength: array like of floats
        The wavelength(s) in nm.
    redshift: float
        The redshift. Must be strictly positive.

    Returns
    -------
    igm_transmission: numpy array of floats
        The intergalactic transmission at each input wavelength.

    """
    from scipy.special import factorial

    n_transitions_low = 10
    n_transitions_max = 31
    gamma = 0.2788  # Gamma(0.5,1) i.e., Gamma(2-beta,1) with beta = 1.5
    n0 = 0.25
    lambda_limit = 91.2  # Lyman limit in nm

    lambda_n = np.empty(n_transitions_max)
    z_n = np.empty((n_transitions_max, len(wavelength)))
    for n in range(2, n_transitions_max):
        lambda_n[n] = lambda_limit / (1. - 1. / float(n * n))
        z_n[n, :] = (wavelength / lambda_n[n]) - 1.

    # From Table 1 in Meiksin (2006), only n >= 3 are relevant.
    # fact has a length equal to n_transitions_low.
    fact = np.array([1., 1., 1., 0.348, 0.179, 0.109, 0.0722, 0.0508, 0.0373,
                     0.0283])

    # First, tau_alpha is the mean Lyman alpha transmitted flux,
    # Here n = 2 => tau_2 = tau_alpha
    tau_n = np.zeros((n_transitions_max, len(wavelength)))
    if redshift <= 4:
        tau_a = 0.00211 * np.power(1. + redshift, 3.7)
        tau_n[2, :] = 0.00211 * np.power(1. + z_n[2, :], 3.7)
    elif redshift > 4:
        tau_a = 0.00058 * np.power(1. + redshift, 4.5)
        tau_n[2, :] = 0.00058 * np.power(1. + z_n[2, :], 4.5)

    # Then, tau_n is the mean optical depth value for transitions
    # n = 3 - 9 -> 1
    for n in range(3, n_transitions_max):
        if n <= 5:
            w = np.where(z_n[n, :] < 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 3.)))
            w = np.where(z_n[n, :] >= 3)
            tau_n[n, w] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, w]), (1. / 6.)))
        elif 5 < n <= 9:
            tau_n[n, :] = (tau_a * fact[n] *
                           np.power(0.25 * (1. + z_n[n, :]), (1. / 3.)))
        else:
            tau_n[n, :] = (tau_n[9, :] * 720. /
                           (float(n) * (float(n * n - 1.))))

    for n in range(2, n_transitions_max):
        # If z_n>=redshift or z_n<0, the photon cannot be absorbed by Lyman n→1
        w = np.where((z_n[n, :] >= redshift) | (z_n[n, :] < 0))
        tau_n[n, w] = 0.

    z_l = wavelength / lambda_limit - 1.
    w = slice(None, np.searchsorted(z_l, redshift))

    tau_l_igm = np.zeros_like(wavelength)
    tau_l_igm[w] = (0.805 * np.power(1. + z_l[w], 3) *
                    (1. / (1. + z_l[w]) - 1. / (1. + redshift)))

    term1 = gamma - np.exp(-1.)

    n = np.arange(n_transitions_low - 1)
    term2 = np.sum(np.power(-1., n) / (factorial(n) * (2 * n - 1)))

    term3 = ((1. + redshift) * np.power(wavelength[w] / lambda_limit, 1.5) -
             np.power(wavelength[w] / lambda_limit, 2.5))

    term4 = np.sum(np.array(
        [((2. * np.power(-1., n) / (factorial(n) * ((6 * n - 5) * (2 * n - 1)))) *
          ((1. + redshift) ** (2.5 - (3 * n)) *
           (wavelength[w] / lambda_limit) ** (3 * n) -
           (wavelength[w] / lambda_limit) ** 2.5))
         for n in np.arange(1, n_transitions_low)]), axis=0)

    tau_l_lls = np.zeros_like(wavelength)
    tau_l_lls[w] = n0 * ((term1 - term2) * term3 - term4)

    # Reset for short wavelength (z_l<0)
    w = slice(None, np.searchsorted(z_l, 0.))

    # Get the normalization factor at z_l=0
    tau_norm_l_igm = np.interp(0, z_l, tau_l_igm)
    tau_norm_l_lls = np.interp(0, z_l, tau_l_lls)

    # Calculate tau_l_igm & tau_l_lls, assuming cross section ~λ^2.75
    # from (O'Meara et al. 2013)
    damp_factor = (z_l[w] + 1.) ** 2.75
    tau_l_igm[w] = tau_norm_l_igm * damp_factor
    tau_l_lls[w] = tau_norm_l_lls * damp_factor

    tau_taun = np.sum(tau_n[2:n_transitions_max, :], axis=0)

    return np.exp(- tau_taun - tau_l_igm - tau_l_lls)


def transmission_Lyman(zObj, lObs):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)
    Args:
        zObj (float): Redshift of object
        lObs (array of float): wavelength grid
    Returns:
        array of float: transmitted flux fraction

    """
    from fastspecfit.util import Lyman_series

    lRF = lObs/(1.+zObj)
    T = np.ones(lObs.size)
    for l in list(Lyman_series.keys()):
        w      = lRF<Lyman_series[l]['line']
        zpix   = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w]  *= np.exp(-tauEff)

    #l912 = lObs <= 911.753 * (1+zObj)
    #if np.any(l912):
    #    T[l912] = 0.0

    return T

def build_ztemplates(filt, restwave, restflux, lyaflux, redshift=2.5, bb_mag=25.0, age=None):
    """Redshifted templates."""

    obswave = restwave * (1 + redshift)
    T = igm_transmission(obswave/10, redshift)
    #T = transmission_Lyman(redshift, obswave)
    obsflux = restflux * T[np.newaxis, :]

    #massnorm = (10.0 / (1e6 * dlum[iz]))**2 / (1.0 + redshift)    

    # normalize to the desired broadband magnitude
    synthmaggies = filt.get_ab_maggies(obsflux, obswave)
    norm = 10**(-0.4 * bb_mag) / synthmaggies['decam2014-g'].value
    
    obsflux = obsflux * norm[:, np.newaxis]
    synthmaggies = filt.get_ab_maggies(obsflux, obswave)
    synthmaggies = np.vstack(synthmaggies.as_array().tolist()).T  

    zlyaflux = lyaflux * norm

    return obswave, obsflux, zlyaflux, synthmaggies

def build_templates(maxage=1e4):
    """Build the templates.

    maxage in Myr

    """
    print('Instantiating the SPS object.')
    sp = fsps.StellarPopulation(compute_vega_mags=False, 
                                add_dust_emission=False,
                                add_neb_emission=True,
                                nebemlineinspec=True,
                                imf_type=1, # Chabrier
                                smooth_velocity=True,
                                #sfh=0,  # SSP
                                sfh=4,  # delayed tau
                                zcontinuous=1, # no interpolation
                                )

    sp.params['logzsol'] = 0.0
    sp.params['tau'] = 0.1  # Gyr
    #sp.params['sigma_smooth'] = 150.0 # [km/s]

    print('Building the SPS spectra.')
    wave, flux = sp.get_spectrum(peraa=True)#, tage=tage)

    lodot = 3.828  # 10^{33} erg/s
    tenpc2 = (10.0 * 3.085678)**2 * 1e3  # 10^{33} cm^2

    flux *= lodot / (4.0 * np.pi * tenpc2) # [erg/s/cm2/A at 10 pc]
    
    lineflux = sp.emline_luminosity * lodot / (4.0 * np.pi * tenpc2) # [erg/s/cm2/A at 10 pc]

    lyaindx = np.argmin(np.abs(sp.emline_wavelengths-1215.0))
    lyaflux = lineflux[:, lyaindx]

    age = 10**sp.log_age / 1e6 # [Myr]
    keep = np.where(age <= maxage)[0]

    flux = flux[keep, :]

    info = Table()
    info['AGE'] = age[keep]               # [Myr]
    info['MSTAR'] = sp.stellar_mass[keep] # [Msun]
    info['SFR'] = sp.sfr[keep]            # [Msun/yr]
    info['SSFR'] = info['SFR'] / info['MSTAR'] # [1/yr]
    info['LYAFLUX'] = lyaflux[keep] # [erg/s/cm2 at 10 pc]

    return wave, flux, info

def main(args):

    # redshift grid
    zgrid = np.arange(args.zmin, args.zmax + args.dz, args.dz)
    nz = len(zgrid)

    # cosmology
    cosmo = TabulatedDESI()
    dlum = cosmo.luminosity_distance(zgrid)

    # filters
    filt = filters.FilterSequence((
        filters.load_filter('desi2-g1.ecsv'), 
        filters.load_filter('desi2-g2.ecsv'),
        filters.load_filter('desi2-g3.ecsv'), 
        filters.load_filter('desi2-g4.ecsv'),
        filters.load_filter('desi2-g5.ecsv'), 
        filters.load_filter('decam2014-g')
    ))
    nfilt = len(filt)
    print(filt.effective_wavelengths.value)

    outfile = args.outroot+'.fits'
    outfile_sed = args.outroot+'-sed.fits'
    if not os.path.isfile(outfile):
        # LAE/LBG templates
        restwave, restflux, info = build_templates()

        hdu1 = fits.PrimaryHDU(restflux)
        hdu2 = fits.ImageHDU(restwave)
        hdu3 = fits.convenience.table_to_hdu(info)

        hdu1.header['EXTNAME'] = 'FLUX'
        hdu2.header['EXTNAME'] = 'WAVE'
        hdu3.header['EXTNAME'] = 'METADATA'

        hx = fits.HDUList([hdu1, hdu2, hdu3])

        print('Writing {}'.format(outfile_sed))
        hx.writeto(outfile_sed, overwrite=True)

        # synthesize photometry on the redshift grid
        zlyaflux, phot = [], []
        for iz, redshift in enumerate(zgrid):
            print('Working on redshift {:.3f}'.format(redshift))
            obswave, obsflux, _zlyaflux, _phot = build_ztemplates(filt, restwave, restflux, info['LYAFLUX'],
                                                                  age=info['AGE'],
                                                                  redshift=redshift, bb_mag=args.bb_mag)

            zlyaflux.append(_zlyaflux)
            phot.append(_phot)

        phot = np.stack(phot, axis=1)
        zlyaflux = np.stack(zlyaflux)
        
        phot = np.vstack((zlyaflux[np.newaxis, :], phot))

        hdu1 = fits.PrimaryHDU(phot.astype('f4'))
        hdu1.header['EXTNAME'] = 'PHOT'
        hdu1.header['ZMIN'] = zgrid[0]
        hdu1.header['ZMAX'] = zgrid[-1]
        hdu1.header['NZ'] = len(zgrid)
        hdu1.header['NFILT'] = nfilt
        for ifilt, name in enumerate(filt.names):
            hdu1.header['FILT{}'.format(ifilt+1)] = name
        hx = fits.HDUList([hdu1])

        print('Writing {}'.format(outfile))
        hx.writeto(outfile, overwrite=True)
        
    # read the results
    phot = fitsio.read(outfile, ext='PHOT')
    restwave = fitsio.read(outfile_sed, ext='WAVE')
    restflux = fitsio.read(outfile_sed, ext='FLUX')
    info = Table(fitsio.read(outfile_sed, ext='METADATA'))

    # SFR and Lya flux vs age
    iz = np.argmin(np.abs(zgrid-2.5))

    fig, ax = plt.subplots()
    leg = ax.plot(info['AGE'], 1e10*info['SFR'], color=colors[0], 
                  label='SFR$\propto te^{-t/\\tau}$')
    ax.set_xlabel('Age [Myr]')
    ax.set_ylabel('Star Formation Rate (Msun/yr) for $10^{10}$ M$_{\odot}$ Galaxy')
    ax.margins(0)
    ax.set_ylim(0, 45)    
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.set_ylim(0, 14)
    ax2.margins(0)
    ax2.set_xscale('log')
    leg2 = ax2.plot(info['AGE'], 1e17 * phot[0, iz, :], color=colors[1], ls='--', label=r'F(Ly$\alpha$)')
    ax2.set_ylabel(r'F(Ly$\alpha$) ($g_{\mathrm{AB}}=25,\, z=2.5$) ($10^{-17}~\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}$)')

    legs = leg + leg2
    labels = [l.get_label() for l in legs]
    ax2.legend(legs, labels, frameon=False, loc='upper right')

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.95)
    fig.savefig('laelbg-sfh.png')

    # SED plot
    redshift = 2.5
    obswave, obsflux, _, _ = build_ztemplates(filt, restwave, restflux, info['LYAFLUX'],
                                              redshift=redshift, bb_mag=args.bb_mag)

    #plt.clf()
    #plt.plot(obswave, transmission_Lyman(redshift, obswave), label='Redrock (from Calura+12)')
    #plt.plot(obswave, igm_transmission(obswave/10, redshift), label='Cigale')
    #plt.xlim(0, 9000)
    #plt.legend()
    #plt.xlabel('Observed-frame Wavelength at z=3')
    #plt.ylabel('Transmission')
    #plt.savefig('junk.png')    

    xlim = np.array([3000, 1e4])
    I = np.where((obswave > xlim[0]) * (obswave < xlim[1]))[0]

    ages = [10, 500, 1e3, 5e3]

    fig, ax = plt.subplots()
    for age in ages:
        im = np.argmin(np.abs(info['AGE']-age))
        if age >= 1e3:
            label = '{:g} Gyr'.format(age/1e3)
        else:
            label = '{:g} Myr'.format(age)
        ax.plot(obswave[I] / 1e4, 1e17*obsflux[im, I], label=label, lw=1)
    ax.set_xlabel('Observed-frame Wavelength ($\mu$m)')
    ax.set_ylabel(r'$F_{\lambda}$ ($10^{-17}~\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}~\AA^{-1}$)')
    ax.set_yscale('log')
    ax.set_xlim(xlim / 1e4)
    ax.set_ylim(1e-4, 1e2)
    ax.margins(x=0)
    ax.legend(loc='upper right', ncols=2)

    for filt1 in filt:
        trim = filt1.wavelength < 6000
        ww = filt1.wavelength[trim]/1e4
        rr = 5e-3*filt1.response[trim]/np.max(filt1.response[trim])+ax.get_ylim()[0]
        if 'decam' in filt1.name:
            plt.plot(ww, rr, color='k', alpha=0.5)
        else:
            plt.plot(ww, rr)
    ax.text(0.05, 0.95, '$g_{{AB}}={:.0f},\, z={:.1f}$'.format(args.bb_mag, redshift),
            ha='left', va='top', transform=ax.transAxes)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    fig.savefig('laelbg-sed.png')

    # color-redshift plot
    im = np.argmin(np.abs(info['AGE']-100))
    iz = np.argmin(np.abs(zgrid-2.5))

    fig, ax = plt.subplots()
    for ifilt in range(nfilt-1):
        col = -2.5 * np.log10(phot[1 + ifilt, :, im]) - args.bb_mag
        ax.plot(zgrid, col, label=filt.names[ifilt])
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Narrowband minus $g$-band Color (AB mag)')
    flux = '{:.3g}'.format(phot[0, iz, im])
    flux = '$'+flux.replace('e-17', '\\times10^{-17}')+'~\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}$'
    ax.text(0.3, 0.9, 'Age=100 Myr\n'+r'F(Ly$\alpha$)={}'.format(flux)+'\n'+r'  at $z=2.5$',
            ha='left', va='top', transform=ax.transAxes)
    ax.margins(0)
    ax.set_ylim(-1.2, 2.2)
    ax.legend()
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    fig.savefig('laelbg-colors-single.png')

    # effect of age
    ifilt = 1
    im = np.argmin(np.abs(info['AGE']-100))

    fig, ax = plt.subplots()
    ycolors, ocolors = [], []
    for imodel in range(len(info)):
        col = -2.5 * np.log10(phot[1 + ifilt, :, imodel]) - args.bb_mag
        #print(info['AGE'][imodel], col[iz])
        if info['AGE'][imodel] < 1400:
            ycolors.append(col)
            #ax.plot(zgrid, col)
        else:
            ocolors.append(col)
    ycolors = np.vstack(ycolors)
    ocolors = np.vstack(ocolors)
    ax.fill_between(zgrid, np.min(ocolors, axis=0), np.max(ocolors, axis=0),
                    label='Age>1.4 Gyr', alpha=0.5)
    ax.fill_between(zgrid, np.min(ycolors, axis=0), np.max(ycolors, axis=0),
                    label='Age<1.4 Gyr', alpha=0.5)

    col = -2.5 * np.log10(phot[1 + ifilt, :, im]) - args.bb_mag
    ax.plot(zgrid, col, label='100 Myr', color='k', lw=2)

    #ax.fill_between(zgrid, np.percentile(ycolors, 25, axis=0), 
    #                np.percentile(ycolors, 75, axis=0))
    #ax.fill_between(zgrid, np.percentile(ocolors, 25, axis=0), 
    #                np.percentile(ocolors, 75, axis=0))
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'desi2-g2 minus $g$-band Color')
    ax.margins(0)
    ax.set_ylim(-1.2, 2.2)
    ax.legend()
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    fig.savefig('laelbg-colors.png')

    # do we need the g-band?
    im = np.argmin(np.abs(info['AGE']-100))
    iz = np.argmin(np.abs(zgrid-2.5))

    fig, ax = plt.subplots()
    for ifilt in range(nfilt-1):
        filtindx = np.delete(np.arange(nfilt), ifilt)
        totmag = -2.5*np.log10(np.sum(phot[1+filtindx, :, im], axis=0))
        col = -2.5 * np.log10(phot[1 + ifilt, :, im]) - totmag
        ax.plot(zgrid, col, label=filt.names[ifilt])
    ax.set_xlabel('Redshift')
    ax.set_ylabel(r'Narrowband$_{i}$ minus $\sum_{j,i\ne j}$ Narrowband$_{j}$ Color')
    flux = '{:.3g}'.format(phot[0, iz, im])
    flux = '$'+flux.replace('e-17', '\\times10^{-17}')+'~\mathrm{erg}~\mathrm{s}^{-1}~\mathrm{cm}^{-2}$'
    ax.text(0.3, 0.9, 'Age=100 Myr', ha='left', va='top', transform=ax.transAxes)
    ax.margins(0)
    ax.set_ylim(0, 3.8)
    ax.legend()
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    fig.savefig('laelbg-colors-nogband.png')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--outroot", help="output root", type=str, default="desi2-laelbg")
    parser.add_argument("--zmin", help="zmin (default=2.0)", type=float, default=2.0)
    parser.add_argument("--zmax", help="zmax (default=3.6)", type=float, default=3.6)
    parser.add_argument("--dz", help="binning in z (default=0.025)", type=float, default=0.025)
    parser.add_argument("--bb_mag", help="broad-band filter magnitude (default=25)", type=float, default=25.0)
    parser.add_argument("--lya_flux", help="Lya flux in erg/s/cm2 (default=6e-17)", type=float, default=6e-17)
    parser.add_argument("--lya_sig", help="Lya Gaussian sigma in A (default=0.8)", type=float, default=0.8)
    parser.add_argument("--lya_skew", help="Lya skew value (default=20)", type=float, default=20)

    #parser.add_argument("--bb_name", help="broad-band filter name (default=decam2014-g)", type=str, default="decam2014-g")
    #parser.add_argument("--mb_wmins", help="comma-separated list of blueward wavelength [A] of the test medium-band filters (default=4000,4000,4000,4000)", type=str, default="4000,4000,4000,4000")
    #parser.add_argument("--mb_wmaxs", help="comma-separated list of redward wavelength [A] of the test medium-band filters (default=4125,4250,4375,4500)", type=str, default="4125,4250,4375,4500")
    args = parser.parse_args()
    
    #for kwargs in args._get_kwargs():
    #    print(kwargs)
    
    main(args)
