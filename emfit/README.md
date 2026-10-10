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
| `smooth-diagnostics` | Per-object smooth-continuum diagnostics from the per-healpix model spectra (and the observed spectra) |
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

# 3. smooth-continuum diagnostics (optional; writes $RUNDIR/smooth-diagnostics.fits)
./smooth-diagnostics --rundir $RUNDIR --mp 128

# 4. compare
./compare-emfit --rundir $RUNDIR
```

`smooth-diagnostics` reads the `MODELS` extension of the per-healpix fastspec
files of every arm (the merged catalogs do not have one) and re-reads the
observed spectra with the fastspecfit I/O, so it needs the same environment as
the fits. Use `--nhealpix 20 --mp 1` for a quick test, `--no-chi2` to skip the
spectra, and `--baseline-dir` (the top-level directory of the per-healpix
iron/v4.0 fastspec files) to add the baseline as an arm.

Each arm is written to `$RUNDIR/<arm>/iron/` and merged into
`$RUNDIR/<arm>/iron/catalogs/fastspec-iron-emfit.fits`. `compare-emfit` picks up
whichever arms exist, so it can be run before all of them finish.

Cost: 40,897 objects in 16,885 healpix files, at about 14 s per file (mean
from job 59592459, which was heavily oversubscribed). The Slurm script runs
the five arms concurrently, one node each, without MPI (`--nompi --mp=128`),
which should take roughly half an hour per arm; it asks for 2 hours. The job
resumes where it stopped if resubmitted. Each arm has its own log in
`$RUNDIR/fastspec-emfit-<arm>-<jobid>.log`.

The first attempt used MPI, and in job 59592459 all 32 `srun` tasks came up
as independent "rank 0" processes, each fitting the whole sample. The cause was
the 26.3 DESI software stack, which broke in a NERSC maintenance;
`mpi-fastspecfit` silently falls back to no MPI when `from mpi4py import MPI`
fails. `etc/fastspecfit-env.sh` on the `smooth-cont` branch now loads 26.9,
where MPI works, but a single process per node is fast enough for this
sample, so the Slurm script stays MPI-free. To check MPI in an interactive
allocation:

```bash
srun -n 4 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size)"
```

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

## Output of `smooth-diagnostics`

`SMOOTHCORR` is the camera median of the smooth continuum over the flux, a
zero-point which any spline reproduces, so it does not distinguish the knot
spacings. `smooth-diagnostics` measures two quantities which do, and
`compare-emfit` summarizes them in `smoothdiag.txt` and `smoothdiag.png` when
`smooth-diagnostics.fits` exists.

- **Cost** (`HA_BUMP`): the smooth continuum integrated over +/-1.5 FWHM
  around H-alpha, after subtracting the straight line which joins its two
  ends. Divided by the EmFit broad H-alpha flux (`FABS_*`), it is the fraction
  of the broad line which the spline absorbed; `FABS_LOST` and `FABS_KEPT` are
  the medians for the gold BL-AGN which lose and keep the broad line of the
  reference arm (`nosmooth`). Objects without an EmFit broad line are assigned
  a FWHM drawn from the gold BL-AGN, so the control sample gives the null
  distribution (`BUMPEW_NL`, as an equivalent width in Angstrom).
- **Benefit** (`DCHI2_{camera}_{sample}`): the decrease in chi2 of the
  line-free pixels per added spline parameter, relative to the next stiffer
  arm, in units of the variance of the residuals. A value of about one means
  that the added knots only fit noise; `FSIG_*` is the fraction of objects in
  which the decrease is significant (3-sigma). The expectation of one is
  approximate, because the knots of two arms are not nested and the spline
  rejects outliers.

Caveats: the model spectra of adjacent cameras are interleaved where the
cameras overlap, so those pixels are left out of the chi2, and `HA_BUMP` is an
average of the two cameras when H-alpha falls there. `HA_BUMP` is undefined
(NaN) when the window runs off the spectrum or more than 20% of it has no
data; isolated masked pixels are interpolated over.

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

- All three scripts have been run at NERSC for the six arms. The
  `compare-emfit` output and `emfit-sample.fits` are also on the laptop, in
  `compare/` and in this directory; the per-arm catalogs are only at NERSC.
- `smooth-diagnostics` and the `smoothdiag` part of `compare-emfit` have
  passed a syntax check only.
- At 200 Angstrom the gold BL-AGN are lost most often when H-alpha is in the r
  camera or the r/z overlap (8.6% at 7400 to 7600 Angstrom, against about 2%
  in the interior of the z camera), and at every knot spacing near the red end
  of the z camera. The cause has not been established.
- In `smoothcorr.png` the legend overlaps the curves in the upper-left panel.
- The plot colors were taken from a validated palette but not re-validated
  (no `node` on the laptop).
- Possible additions: recovery in bins of observed H-alpha wavelength;
  comparison of the fastspecfit smooth continuum at H-alpha with EmFit's
  `NII_HA_CONTINUUM`; H-beta broad-line comparison (the columns are already in
  both files).
- Side finding in the fastspecfit repo, not acted on: `doc/changes.rst` on
  `smooth-cont` still advertises `--smooth-window` and `--smooth-step`, but the
  command line now exposes `--smooth-knot-spacing`.
