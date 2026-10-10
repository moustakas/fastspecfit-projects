"""
Shared definitions for the fastspecfit-vs-EmFit comparison scripts
(``build-emfit-sample`` and ``compare-emfit``).

"""
import numpy as np

SIGMA2FWHM = 2. * np.sqrt(2. * np.log(2.))

SAMPLEFILE = 'emfit-sample.fits'
BASELINEFILE = 'fastspec-iron-v4.0-emfit.fits'
BASELINE = 'v4.0'
DIAGFILE = 'smooth-diagnostics.fits' # written by smooth-diagnostics

C_LIGHT = 299792.458 # [km/s]
HALPHA_WAVE = 6564.60 # vacuum [Angstrom]

# EmFit columns copied into the reference catalog.
EMFIT_IDCOLS = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'TARGET_RA', 'TARGET_DEC',
                'Z', 'PROB_BROAD']
EMFIT_COMPONENTS = ['HB_N', 'HB_OUT', 'HB_B', 'OIII5007', 'OIII5007_OUT',
                    'NII6583', 'NII6583_OUT', 'HA_N', 'HA_OUT', 'HA_B',
                    'SII6716', 'SII6716_OUT', 'SII6731', 'SII6731_OUT']
EMFIT_SUFFIXES = ['AMPLITUDE', 'FLUX', 'FLUX_ERR', 'SIGMA', 'SIGMA_ERR', 'SIGMA_FLAG']
EMFIT_WINDOWS = ['HB', 'OIII', 'NII_HA', 'SII']
EMFIT_WINDOW_SUFFIXES = ['CONTINUUM', 'CONTINUUM_ERR', 'NOISE', 'NDOF', 'RCHI2']
EMFIT_REFCOLS = (EMFIT_IDCOLS +
                 [f'{comp}_{suffix}' for comp in EMFIT_COMPONENTS for suffix in EMFIT_SUFFIXES] +
                 [f'{win}_{suffix}' for win in EMFIT_WINDOWS for suffix in EMFIT_WINDOW_SUFFIXES] +
                 ['HB_OIII_NDOF', 'HB_OIII_RCHI2', 'NII_HA_SII_NDOF', 'NII_HA_SII_RCHI2',
                  'OIII_DBL_FLAG', 'SII_DBL_FLAG'])

# The double-peak flag which governs each narrow EmFit component.
EMFIT_DBLFLAG = {'HB_N': 'SII_DBL_FLAG', 'HA_N': 'SII_DBL_FLAG', 'NII6583': 'SII_DBL_FLAG',
                 'SII6716': 'SII_DBL_FLAG', 'SII6731': 'SII_DBL_FLAG',
                 'OIII5007': 'OIII_DBL_FLAG'}

# Columns of the Pucha+26 BL-AGN (Zenodo) catalog copied into the reference
# catalog, with the fill value used for objects not in that catalog.
BLAGN_COLS = {'VI_FLAG': -1, 'EBL_AGN': False, 'HALPHA_BROAD_FWHM': 0.,
              'LOG_MBH': 0., 'LOGM': 0., 'LSDR9_MORPHOLOGY': '', 'NII_BPT': ''}

# fastspec columns used in the comparison.
FAST_METACOLS = ['TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX', 'Z']
FAST_LINES = ['HALPHA', 'HALPHA_BROAD', 'HBETA', 'HBETA_BROAD', 'NII_6584',
              'OIII_5007', 'SII_6716', 'SII_6731']
FAST_LINE_SUFFIXES = ['_FLUX', '_FLUX_IVAR', '_SIGMA', '_SIGMA_IVAR', '_AMP',
                      '_AMP_IVAR', '_CONT', '_CONT_IVAR', '_EW']
FAST_CAMERAS = ['B', 'R', 'Z']
FAST_COLS = ([f'SMOOTHCORR_{cam}' for cam in FAST_CAMERAS] +
             [f'SNR_{cam}' for cam in FAST_CAMERAS] +
             ['DELTA_LINECHI2', 'DELTA_LINENDOF'] +
             [f'{line}{suffix}' for line in FAST_LINES for suffix in FAST_LINE_SUFFIXES])


def arm_sortkey(arm):
    """Sort key which orders the arms as: baseline, knots (in order of
    increasing spacing), everything else.

    """
    if arm == BASELINE:
        return (0, 0.)
    elif arm.startswith('knots'):
        return (1, float(arm[5:]))
    return (2, 0.)


def strip(strings):
    """Convert a (possibly padded, possibly bytes) string column to `str`.

    """
    return np.char.strip(np.asarray(strings).astype(str))


def snr(flux, ferr):
    """Signal-to-noise ratio, set to zero where the uncertainty is not positive.

    """
    out = np.zeros(len(flux))
    I = ferr > 0.
    out[I] = flux[I] / ferr[I]
    return out


def emfit_narrow_flux(cat, comp):
    """Total narrow-line flux of an EmFit component.

    Following Pucha+26, the primary and secondary components are summed (and
    their uncertainties added in quadrature) when the line is double-peaked;
    otherwise only the primary component is used.

    Parameters
    ----------
    cat : :class:`numpy.ndarray` or :class:`astropy.table.Table`
        EmFit catalog.
    comp : str
        Narrow component, e.g., ``HA_N`` or ``OIII5007``.

    Returns
    -------
    flux, ferr : :class:`numpy.ndarray`
        Flux and its uncertainty in 1e-17 erg/s/cm2.

    """
    out = comp.replace('_N', '_OUT') if comp.endswith('_N') else f'{comp}_OUT'
    dbl = np.asarray(cat[EMFIT_DBLFLAG[comp]]).astype(bool)
    flux = np.array(cat[f'{comp}_FLUX'], dtype='f8')
    ferr = np.array(cat[f'{comp}_FLUX_ERR'], dtype='f8')
    flux[dbl] += cat[f'{out}_FLUX'][dbl]
    ferr[dbl] = np.hypot(ferr[dbl], cat[f'{out}_FLUX_ERR'][dbl])
    return flux, ferr


def match_targetid(reftid, tid):
    """Match two lists of TARGETIDs.

    Parameters
    ----------
    reftid : :class:`numpy.ndarray`
        Unique reference TARGETIDs.
    tid : :class:`numpy.ndarray`
        TARGETIDs to look up in ``reftid``.

    Returns
    -------
    :class:`numpy.ndarray`
        For each element of ``tid``, its index in ``reftid`` or -1.

    """
    srt = np.argsort(reftid)
    idx = np.clip(np.searchsorted(reftid, tid, sorter=srt), 0, len(reftid) - 1)
    idx = srt[idx]
    idx[reftid[idx] != tid] = -1
    return idx
