#!/usr/bin/env python
"""Fit the custom M31 reductions.

time /global/u2/i/ioannis/code/desihub/fastspecfit-projects/m31/fastspecfit-m31 --preprocess
time /global/u2/i/ioannis/code/desihub/fastspecfit-projects/m31/fastspecfit-m31 --mp 32 --targetids custom
time /global/u2/i/ioannis/code/desihub/fastspecfit-projects/m31/fastspecfit-m31 --mp 32 --makeqa
time /global/u2/i/ioannis/code/desihub/fastspecfit-projects/m31/fastspecfit-m31 --makehtml

rsync -auvP /global/cfs/cdirs/desi/users/ioannis/fastspecfit/m31/fujilupe ~/cosmo-www/m31/


As you probably saw, our initial paper on M31 is done, but this does not include
any analyses of sources that are in the disk of the galaxy. Several of these are
targets which have both stellar spectra (sometimes clusters or confused regions)
and emsision lines. As we have discussed before, it would be really interesting
to measure the emission line ratios in some systematic way, be able to separate
sources by their ionization (i.e., PNe and HII) and investigate their kinematics
and gas-phase metallicities.

/global/homes/k/koposov/desi_koposov/m31_processing_scripts/reduction_2205/data/coadd_m31_all_220521.fits - coadd file
rvtab_coadd_m31_all_220521.fits - rvtab file

"""
import pdb # for debugging

import os, time, subprocess
import numpy as np
import fitsio
from glob import glob
from astropy.table import Table
from fastspecfit.util import C_LIGHT

from desiutil.log import get_logger
log = get_logger()

def preprocess(outdir):
    """Can't use the Docker container for pre-processing because we do not have
    redrock:
   
    fastspecfit is aliased to `source $IMPY_DIR/bin/fastspecfit-env-nersc'
    
    """
    from redrock.external.desi import write_zbest
    from desispec.io import write_spectra, read_spectra

    datadir = '/global/homes/k/koposov/desi_koposov/m31_processing_scripts/reduction_2205/data'

    # read the updated redshift catalog
    rvcat = Table(fitsio.read(os.path.join(outdir, 'M31_ALLTILES_RVTAB_GOOD_2022may21.fits.gz')))
    targetid = rvcat['TARGETID'].data
    znew = rvcat['VRAD_BEST'].data / C_LIGHT

    # read the redrock and coadd catalog
    coaddfile = os.path.join(datadir, 'coadd_m31_all_220521.fits')
    redrockfile = os.path.join(datadir, 'redrock_m31_all_220521.fits')

    outcoaddfile = os.path.join(outdir, 'coadd-m31_rvtab_good_220521.fits')
    outredrockfile = os.path.join(outdir, 'redrock-m31_rvtab_good_220521.fits')

    redhdr = fitsio.read_header(redrockfile)
    zbest = Table.read(redrockfile, 'REDSHIFTS')
    fibermap = Table.read(redrockfile, 'FIBERMAP')
    expfibermap = Table.read(redrockfile, 'EXP_FIBERMAP')
    tsnr2 = Table.read(redrockfile, 'TSNR2')

    I = np.hstack([np.where(tid == zbest['TARGETID'])[0] for tid in targetid])

    zbest = zbest[I]
    fibermap = fibermap[I]
    expfibermap = expfibermap[I]
    tsnr2 = tsnr2[I]
    assert(np.all(zbest['TARGETID'] == targetid))

    zbest['Z_ORIG'] = zbest['Z']
    zbest['Z'] = znew

    spechdr = fitsio.read_header(coaddfile)
    spec = read_spectra(coaddfile).select(targets=targetid)
    assert(np.all(spec.fibermap['TARGETID'] == targetid))
    
    # update the headers so things work with fastspecfit
    redhdr['SPGRP'] = 'healpix'
    redhdr['SPGRPVAL'] = 0
    redhdr['SURVEY'] = 'custom'
    redhdr['PROGRAM'] = 'm31'
    redhdr['SPECPROD'] = 'custom'
    
    spechdr['SPGRP'] = 'healpix'
    spechdr['SPGRPVAL'] = 0
    spechdr['SURVEY'] = 'custom'
    spechdr['PROGRAM'] = 'm31'
    spechdr['SPECPROD'] = 'custom'
    spec.meta = spechdr

    print('Writing {}'.format(outcoaddfile))
    write_spectra(outcoaddfile, spec)
    
    archetype_version = None
    template_version = {redhdr['TEMNAM{:02d}'.format(nn)]: redhdr['TEMVER{:02d}'.format(nn)] for nn in np.arange(10)}

    print('Writing {}'.format(outredrockfile))
    write_zbest(outredrockfile, zbest, fibermap, expfibermap, tsnr2,
                template_version, archetype_version, spec_header=redhdr)

    pdb.set_trace()
    
def main():
    """Main wrapper on fastspec.

    """
    import argparse    
    from fastspecfit.mpi import plan
    from fastspecfit.continuum import ContinuumFit
    from fastspecfit.emlines import EMLineFit
    from fastspecfit.io import DESISpectra, write_fastspecfit, read_fastspecfit
    from fastspecfit.fastspecfit import _fastspec_one, fastspec_one, _desiqa_one, desiqa_one
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mp', type=int, default=1, help='Number of multiprocessing processes per MPI rank or node.')
    parser.add_argument('-n', '--ntargets', type=int, help='Number of targets to process in each file.')
    parser.add_argument('--targetids', type=str, default=None, help='Comma-separated list of TARGETIDs to process.')
    
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the files.')
    parser.add_argument('--makeqa', action='store_true', help='Build QA in parallel.')
    parser.add_argument('--makehtml', action='store_true', help='Build the HTML page.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite any existing output files.')

    args = parser.parse_args()

    # project parameters
    outdir = '/global/cfs/cdirs/desi/users/ioannis/fastspecfit/m31/fujilupe'
    redrockfile = os.path.join(outdir, 'redrock-m31_rvtab_good_220521.fits')
    fastfitfile = os.path.join(outdir, 'fastspec-m31_rvtab_good_220521-test.fits')
    
    qadir = os.path.join(outdir, 'qa')
    if not os.path.isdir(qadir):
        os.makedirs(qadir, exist_ok=True)

    if args.preprocess or not os.path.isfile(redrockfile):
        preprocess(outdir)

    samplefile = redrockfile
    sample = Table(fitsio.read(samplefile))
    sample['SURVEY'] = 'custom'
    sample['PROGRAM'] = 'm31'
    sample['HEALPIX'] = 0

    # select a subset of targets
    if args.targetids:
        if args.targetids == 'custom':
            targetids = np.loadtxt(os.path.join(outdir, 'for_john_clusters_h2pn_targetids.txt'), dtype=int)
        else:
            targetids = np.array([int(x) for x in args.targetids.split(',')])
        sample = sample[np.isin(sample['TARGETID'], targetids)]
    else:
        targetids = None

    if args.makehtml:
        fastfit, metadata, coadd_type, _ = read_fastspecfit(fastfitfile)

        pngfiles = glob(os.path.join(qadir, '*.png'))
        if len(pngfiles) == 0:
            print('No PNG files found in {}'.format(qadir))
            return

        indexfile = os.path.join(qadir, 'index.html')
        print('Writing {}'.format(indexfile))
        with open(indexfile, 'w') as html:
            html.write('<html><body>\n')
            html.write('<style type="text/css">\n')
            html.write('table, td, th {padding: 5px; text-align: center; border: 1px solid black}\n')
            html.write('</style>\n')

            html.write('<h1>M31 Project</h1>\n')
            html.write('<h3><a href="../{}">Download fastspec catalog</a> (<a href="https://fastspecfit.readthedocs.io/en/latest/fastspec.html">Data Model</a>)</h3>\n'.format(os.path.basename(fastfitfile)))

            for survey in set(metadata['SURVEY']):
                I = survey == metadata['SURVEY']
                for program in set(metadata['PROGRAM'][I]):
                    J = program == metadata['PROGRAM'][I]

                    html.write('<h2>{} - {}</h2>\n'.format(survey, program))
                    html.write('<table>\n')
                    for S in metadata[I][J]:
                        pngfile = os.path.join(qadir, 'fastspec-{}-{}-{}-{}.png'.format(S['SURVEY'], S['PROGRAM'], S['HEALPIX'], S['TARGETID']))
                        if os.path.isfile(pngfile):
                            #print(survey, program, pngfile)
                            html.write('<tr width="90%"><td colspan="4"><a href="{0}"><img src="{0}" height="auto" width="1024px"></a></td></tr>\n'.format(os.path.basename(pngfile)))
                            html.write('<tr width="90%">')
                            html.write('<td>{}</td>'.format(S['TARGETID']))
                            html.write('<td>{}</td>'.format(S['SURVEY']))
                            html.write('<td>{}</td>'.format(S['PROGRAM']))
                            html.write('<td>{}</td>'.format(S['HEALPIX']))
                            html.write('</tr>\n')
                        else:
                            pdb.set_trace()
                    html.write('</table>\n')
                    html.write('<br />\n')

        return

    Spec = DESISpectra()
    CFit = ContinuumFit()
    EMFit = EMLineFit()

    if args.makeqa:
        fastfit, metadata, coadd_type, _ = read_fastspecfit(fastfitfile)
        targetids = metadata['TARGETID'].data

        Spec.select(redrockfiles=redrockfile, targetids=targetids, use_quasarnet=False)
        data = Spec.read_and_unpack(CFit, fastphot=False, synthphot=True, remember_coadd=True)
        
        indx = np.arange(len(data))
        qaargs = [(CFit, EMFit, data[igal], fastfit[indx[igal]], metadata[indx[igal]],
                   coadd_type, False, qadir, None) for igal in np.arange(len(indx))]                

        if args.mp > 1:
            import multiprocessing
            with multiprocessing.Pool(args.mp) as P:
                P.map(_desiqa_one, qaargs)
        else:
            [desiqa_one(*_qaargs) for _qaargs in qaargs]
    else:
        Spec.select(redrockfiles=redrockfile, targetids=targetids, zmin=-0.1,
                    use_quasarnet=False, ntargets=args.ntargets)
        data = Spec.read_and_unpack(CFit, fastphot=False, synthphot=True, remember_coadd=True)

        out, meta = Spec.init_output(CFit=CFit, EMFit=EMFit, fastphot=False)
        
        # Fit in parallel
        t0 = time.time()
        fitargs = [(iobj, data[iobj], out[iobj], meta[iobj], CFit, EMFit, False, False) # verbose and broadlinefit
                   for iobj in np.arange(Spec.ntargets)]
        if args.mp > 1:
            import multiprocessing
            with multiprocessing.Pool(args.mp) as P:
                _out = P.map(_fastspec_one, fitargs)
        else:
            _out = [fastspec_one(*_fitargs) for _fitargs in fitargs]
        _out = list(zip(*_out))
        out = Table(np.hstack(_out[0]))
        meta = Table(np.hstack(_out[1]))
        log.info('Fitting everything took: {:.2f} sec'.format(time.time()-t0))

        # Write out.
        write_fastspecfit(out, meta, outfile=fastfitfile, specprod=Spec.specprod,
                          coadd_type=Spec.coadd_type, fastphot=False)

if __name__ == '__main__':
    main()
