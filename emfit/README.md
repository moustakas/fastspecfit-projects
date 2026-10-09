# Testing the fastspecfit smooth continuum against EmFit (DR1)

The `smooth-cont` branch of fastspecfit replaces the sliding-window smooth
continuum with a per-camera fixed-knot spline (`--smooth-knot-spacing`, default
200 Angstrom). The smooth continuum should be flexible enough to absorb genuine
flux-calibration errors without subtracting broad Balmer lines, especially the
relatively narrow broad lines of low-mass black holes. The default is meant to
be good enough for most objects, not tuned to low-mass AGN.

Here we fit one sample with several knot spacings and compare against the
EmFit DR1 catalog of Pucha+25 and Pucha+26 (arXiv:2606.02699) and against the
iron/v4.0 FastSpecFit VAC (tag 3.6.1, the old smooth continuum).

Exact agreement with EmFit is not expected: EmFit works in the rest frame on a
resampled spectrum, subtracts the stellar continuum of the public DR1
FastSpecFit VAC, fits a local polynomial continuum in each window, and allows
outflow and double-peaked narrow components.

## Files

| File | Purpose |
|---|---|
| `build-emfit-sample` | Select the sample; write the samplefile / reference catalog and the iron/v4.0 baseline |
| `fastspec-emfit.slurm` | Fit the sample once per arm with `mpi-fastspecfit --samplefile`, and merge |
| `compare-emfit` | Summary statistics, figures, and lists of changed objects |
| `emfit_util.py` | Column lists and helpers shared by the two Python scripts |

## Inputs

| What | Where |
|---|---|
| EmFit v2.3.1 (17 GB; HDU `EMFIT`; 7,378,347 rows; 305 columns) | NERSC: `/global/cfs/cdirs/desi/public/dr1/vac/dr1/emfit/`; laptop: `~/Downloads/emfit-dr1-v2.3.1.fits` |
| Pucha+26 BL-AGN catalog, `desi-dr1-all-bl-agn.fits` | https://doi.org/10.5281/zenodo.21539891; laptop: `~/Downloads/21539891.zip`. Copy the FITS file to NERSC |
| iron/v4.0 merged catalogs | NERSC: `/dvs_ro/cfs/cdirs/desi/vac/dr1/fastspecfit/iron/v4.0/catalogs/` |
| fastspecfit, branch `smooth-cont` | NERSC: `/global/common/software/desi/users/ioannis/fastspecfit` (`fastspecfit_dir` in the Slurm script) |

## Sample

`emfit-sample.fits` holds three subsamples (column `SAMPLE`):

- **`bl`**: all 17,949 BL-AGN of the Pucha+26 Zenodo catalog. These pass the
  full Pucha+26 selection (photometry, CIGALE stellar mass, line-emitting,
  `PROB_BROAD` >= 80, broad H-alpha flux and width S/N >= 3, AoN >= 2,
  [NII]-BPT AGN or composite) and the stacking / visual vetting of the
  FWHM < 1000 km/s candidates. `VI_FLAG = 0` (17,788; the "gold" sample) are
  confident and `VI_FLAG = 1` (161) are tentative. 13,026 are default-mode fits
  and 4,923 are EBL-mode fits.
- **`nl`**: a control sample of the same size. The pool is EmFit line-emitting
  galaxies (S/N >= 3 in [OIII], H-alpha, [NII]; S/N >= 1 and AoN >= 1 in
  H-beta) with `PROB_BROAD = 0` (about 2.06 million), from which we draw as
  many objects as there are BL-AGN in each bin of redshift (0.025) and log
  narrow H-alpha S/N (0.25 dex). The photometry and stellar-mass cuts of
  Pucha+26 are not applied to the control.
- **`calib`**: 5,000 EmFit objects drawn at random from the top 1% of
  max(|`SMOOTHCORR_B`|, |`SMOOTHCORR_R`|, |`SMOOTHCORR_Z`|) in iron/v4.0, i.e.,
  where the old smooth continuum made a large correction.

Width distribution of the `bl` sample (EmFit FWHM of broad H-alpha):

| FWHM [km/s] | N | Notes |
|---|---|---|
| 327 to 589 | 142 | Below the fastspecfit floor; 86 are `VI_FLAG = 1` |
| 589 to 1000 | 684 | 75 are `VI_FLAG = 1` |
| 1000 to 2000 | 5,923 | |
| 2000 to 4000 | 7,141 | 1,515 EBL |
| above 4000 | 4,059 | 3,408 EBL |

792 objects have log M_BH < 6 and 434 have log M* <= 9.5. The median redshift is
0.26, so H-alpha is typically at about 8270 Angstrom in the z camera.

## Arms

| Arm | Setting |
|---|---|
| `v4.0` | iron/v4.0 baseline (read from the VAC; old smooth continuum) |
| `knots100` | `--smooth-knot-spacing=100` |
| `knots200` | `--smooth-knot-spacing=200` (branch default) |
| `knots400` | `--smooth-knot-spacing=400` |
| `knots800` | `--smooth-knot-spacing=800` |
| `nosmooth` | `--no-smooth-continuum` |

Why this grid: at z = 0.26 one Angstrom is about 36 km/s at H-alpha. A broad
line with FWHM of 600 to 1500 km/s (the low-mass regime) is 17 to 42 Angstrom
wide at half maximum and about 40 to 105 Angstrom at its base; the typical
BL-AGN (FWHM of about 1500 to 4600 km/s) is 105 to 315 Angstrom at its base. So
100 Angstrom knots are comparable to the full extent of the narrowest broad
lines, 200 Angstrom to that of the typical ones, and 400 and 800 Angstrom are
stiffer than nearly all of them. The z camera spans about 2400 Angstrom, i.e.,
24, 12, 6, and 3 spline pieces. The minimum number of pixels between knots
(15) is left at its default.

## Recipe (all at NERSC)

```bash
cd $HOME/code/fastspecfit-projects/emfit   # this directory
source /global/common/software/desi/users/ioannis/fastspecfit/etc/fastspecfit-env.sh
export PYTHONPATH=/global/common/software/desi/users/ioannis/fastspecfit/py:$PYTHONPATH   # smooth-cont checkout
export RUNDIR=$PSCRATCH/fastspecfit/emfit

# 1. sample + baseline (reads every iron/v4.0 merged catalog once)
./build-emfit-sample \
    --emfitfile /global/cfs/cdirs/desi/public/dr1/vac/dr1/emfit/v2.3/emfit-dr1-v2.3.1.fits \
    --blagnfile $RUNDIR/desi-dr1-all-bl-agn.fits \
    --vacdir /dvs_ro/cfs/cdirs/desi/vac/dr1/fastspecfit/iron/v4.0/catalogs \
    --outdir $RUNDIR

# 2. fit and merge every arm (edit the variables at the top first)
sbatch fastspec-emfit.slurm

# 3. compare
./compare-emfit --rundir $RUNDIR
```

Each arm is written to `$RUNDIR/<arm>/iron/` and merged into
`$RUNDIR/<arm>/iron/catalogs/fastspec-iron-emfit.fits`. `compare-emfit` picks up
whichever arms exist, so it can be run before all of them finish.

Cost: about 41,000 objects spread over roughly 25,000 healpix files, so
per-file I/O is comparable to the fitting time (about 8 s/object/core for
bright-time targets). The Slurm script asks for 4 nodes and 4 hours for the
five arms; this is an estimate, and the job resumes where it stopped if
resubmitted.

## Output of `compare-emfit`

- `summary.txt`, one row per arm:
  - `REC_*`: fraction of BL-AGN in which fastspecfit has a broad H-alpha line
    (flux > 0 and S/N >= `--minsnr-broad`, default 3): all `bl`, gold, gold in
    three FWHM ranges, and gold with log M_BH < 6.
  - `FALSEPOS_NL`: the same fraction in the control sample.
  - `DLOGFWHM`, `DLOGFBROAD`: median (and `_NMAD` scatter) of log10
    fastspecfit / EmFit for the broad H-alpha FWHM and flux of the recovered
    gold sample.
  - `DLOGHA_*`, `DLOGNII_*`: the same for the narrow H-alpha and [NII] 6584
    fluxes, for the gold and control samples. EmFit primary and secondary
    components are summed where the line is double-peaked.
  - `SMOOTHCORR_{B,R,Z}_*`: median |smooth-continuum correction| in percent.
- `recovery.png`: recovery rate against EmFit FWHM, and the false-positive rate.
- `fwhm.png`: fastspecfit against EmFit broad H-alpha FWHM for the gold sample.
- `smoothcorr.png`: cumulative distributions of |`SMOOTHCORR`| by sample and
  camera.
- `changed-<arm>.txt`: gold BL-AGN `lost` or `gained` and control objects with
  a `spurious` broad line relative to the baseline, for inspection with
  `fastqa`.

## Things to keep in mind when reading the results

- **fastspecfit cannot return a broad line narrower than FWHM 589 km/s.**
  `emline_specfit` drops the broad model when the broad H-alpha sigma is below
  `minsigma_balmer_broad = 250` km/s (it also requires delta-chi2 > delta-ndof,
  broad sigma > narrow sigma, and broad Balmer S/N > 2.5). EmFit's floor is FWHM
  300 km/s, so the lowest FWHM bin (142 objects) is unrecoverable by
  construction and is shown separately.
- **Low-FWHM EmFit broad lines are uncertain.** Pucha+26 (Appendices A.3 and
  B) removed most FWHM < 1000 km/s candidates as missed outflows or poor fits;
  those that remain are in the Zenodo catalog, with `VI_FLAG` marking the
  tentative ones.
- **Outflows.** 876 of the BL-AGN have an H-alpha `_OUT` component and 8,583
  an [OIII] one; fastspecfit has no outflow component, so some of that flux
  can end up in its broad line. The reference catalog keeps the `_OUT` columns
  so these can be separated.
- **EmFit conventions.** `*_SIGMA` is in km/s and corrected for instrumental
  resolution (`HA_B_SIGMA_FLAG = 0` for the whole `bl` sample); fluxes are in
  1e-17 erg/s/cm2; wavelengths are vacuum; `Z` is Redrock with the QuasarNet
  corrections, the fastspecfit default. EBL-mode fits have
  `NII_HA_SII_NDOF > 0` (identical to `EBL_AGN` in the Zenodo catalog). The
  broad component is a single Gaussian.
- **The baseline differs from the branch by more than the smooth continuum**:
  different Monte Carlo seeds (per-file versus sample runs) and the
  emission-line updates of fastspecfit PR #289. Neither should matter near
  H-alpha, but the `knots*` arms compared against each other are the clean
  test.
- **The `calib` sample has no ground truth.** It shows how much correction
  each knot spacing still makes where the old algorithm made a large one; the
  judgment of whether that correction is real needs `fastqa` on a subset.

## Not yet done

- Nothing here has been run. The Python scripts pass a syntax check only.
- The plot colors were taken from a validated palette but not re-validated
  (no `node` on the laptop).
- Possible additions: recovery in bins of the smooth correction itself;
  comparison of the fastspecfit smooth continuum at H-alpha with EmFit's
  `NII_HA_CONTINUUM`; H-beta broad-line comparison (the columns are already in
  both files).
- Side finding in the fastspecfit repo, not acted on: `doc/changes.rst` on
  `smooth-cont` still advertises `--smooth-window` and `--smooth-step`, but the
  command line now exposes `--smooth-knot-spacing`.
