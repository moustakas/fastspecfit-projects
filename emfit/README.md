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
| `fastqa-emfit` | Select a few objects per open question from the diagnostics and build their `fastqa` figures, once per arm |
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
| `knots800` | `--smooth-knot-spacing=800` (branch default since 2026-10-10), current code |
| `nosmooth` | `--no-smooth-continuum` |
| `old-knots800` | The 800 Angstrom fit of 2026-10-09, i.e., before the short runs of pixels at the ends of a camera were dropped (fastspecfit 1fbef9b) |
| `prebatch-knots800` | 800 Angstrom, before the Monte Carlo realizations were fit at once with the rejected pixels of the nominal fit (fastspecfit 2b7acc7) |

The `knots100`, `knots200`, `knots400`, and `nosmooth` arms were fit on
2026-10-09 and not since, so they also predate 1fbef9b (which does not matter
for `nosmooth`). The catalogs fit before 2026-10-10 have
`SMOOTHCORR_{B,R,Z}`, and the later ones (`knots800` and `prebatch-knots800`)
have `SMOOTHRMS_{B,R,Z}`, `DELTA_SMOOTHCHI2_{B,R,Z}`, and
`DELTA_SMOOTHNDOF_{B,R,Z}` in their place (fastspecfit cab7c44).

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
export PATH=/global/common/software/desi/users/ioannis/fastspecfit/bin:$PATH   # smooth-cont checkout
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

# 5. figures for visual inspection (writes compare/fastqa-targets.txt and $RUNDIR/fastqa/)
./fastqa-emfit --rundir $RUNDIR --mp 32
```

`smooth-diagnostics` reads the `MODELS` extension of the per-healpix fastspec
files of every arm (the merged catalogs do not have one) and re-reads the
observed spectra with the fastspecfit I/O, so it needs the same environment as
the fits. Use `--nhealpix 20 --mp 1` for a quick test, `--no-chi2` to skip the
spectra, and `--baseline-dir` (the top-level directory of the per-healpix
iron/v4.0 fastspec files) to add the baseline as an arm.

The healpix files are processed in chunks of `--chunksize` (default 500), and
each chunk is written to `$RUNDIR/smooth-diagnostics-chunks/` when it is done
(and its timing logged), so repeating the same command resumes an interrupted
run. A checkpoint written with different arms, objects, `--no-chi2`,
`--minsnr-broad`, or `--seed` is an error; use `--overwrite` to start over.
Nothing is checkpointed with `--nhealpix`.

Three runs on 2026-10-10 and 2026-10-11 stopped making progress within the
first chunk, with no error, because workers died and `multiprocessing.Pool`
waits forever for the result of a dead worker. The script's own pool
(`HealpixPool`) notices a dead worker, starts over with half as many workers,
and repeats the unfinished healpix files. It also lists the healpix files in
progress if none finishes in 10 minutes, so a file which blocks would be named
in the log.

On 2026-10-11 the workers died with 128, 64, and 32 workers, all within about
70 s of the start, and the run went through with 16 (about 2 minutes per
chunk). Why they die is not known. It is not the memory in steady state (0.5
GB per worker, with `DESISpectra.read` now reading only the requested targets
of each coadd) and it is not the number of threads (`OMP_NUM_THREADS=1`, one
thread per worker). The messages which the workers printed when they died
have not been looked at; until they are, use `--mp 16`.

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
  - `SMOOTHCORR_{B,R,Z}_*`: median |smooth-continuum correction| in percent
    (NaN for the arms whose catalog does not have `SMOOTHCORR`).
  - `SMOOTHRMS_{B,R,Z}_*`: median rms of the smooth continuum relative to the
    median stellar continuum, in percent (NaN for the arms whose catalog does
    not have `SMOOTHRMS`). The two statistics are not comparable.
- `recovery.png`: recovery rate against EmFit FWHM, and the false-positive rate.
- `fwhm.png`: fastspecfit against EmFit broad H-alpha FWHM for the gold sample.
- `smoothcorr.png`: cumulative distributions of |`SMOOTHCORR`| by sample and
  camera, for the arms which have it.
- `smoothrms.png`: the same for `SMOOTHRMS`.
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

## Results (2026-10-09)

All numbers are from `compare/summary.txt`, `compare/changed-<arm>.txt`, and
`emfit-sample.fits`; the figures are in `compare/`. The fits were run with
`--minsnr-broad` at its default of 3.

In this section and the next, `knots800` is the fit of 2026-10-09, which is
now the `old-knots800` arm. The current code is in "800 Angstrom with the
current code" below.

### Recovery of the gold BL-AGN and agreement with EmFit

| Arm | Recovered | FWHM < 1000 | 1000 to 2000 | FWHM > 2000 | log M_BH < 6 | Control with a broad line | FWHM scatter [dex] | Lost / gained vs v4.0 |
|---|---|---|---|---|---|---|---|---|
| `v4.0` | 88.6% | 65.0% | 80.6% | 94.2% | 54.5% | 1.04% | 0.030 | |
| `knots100` | 77.2% | 55.6% | 71.4% | 81.5% | 45.4% | 0.94% | 0.059 | 2219 / 198 |
| `knots200` | 86.1% | 62.3% | 79.0% | 91.4% | 50.6% | 1.14% | 0.041 | 635 / 205 |
| `knots400` | 87.2% | 64.2% | 79.8% | 92.5% | 52.8% | 1.04% | 0.032 | 374 / 132 |
| `knots800` | 88.4% | 65.6% | 80.5% | 94.0% | 55.3% | 1.07% | 0.029 | 143 / 119 |
| `nosmooth` | 88.8% | 66.2% | 80.8% | 94.3% | 55.9% | 1.11% | 0.034 | 133 / 170 |

- Recovery rises monotonically with the knot spacing. The 200 Angstrom default
  loses 2.4 points relative to iron/v4.0, and 100 Angstrom loses 11.
- `nosmooth` against iron/v4.0 (133 lost, 170 gained) is the floor set by the
  Monte Carlo seeds, PR #289, and the S/N = 3 threshold. `knots800` is at that
  floor; `knots400` and `knots200` are about 240 and 500 lost objects above it.
- The false-positive rate does not discriminate: with about 18,000 control
  objects the Poisson uncertainty is about 0.08 points.
- Flexible knots add scatter. The [NII] scatter of the control sample
  (`DLOGNII_NL_NMAD`) is 0.036, 0.038, 0.046, and 0.062 dex at 800, 400, 200,
  and 100 Angstrom.
- With no smooth continuum the broad line is biased high relative to EmFit
  (+0.036 dex in FWHM and +0.031 dex in flux). The offsets are smallest at
  400 Angstrom (+0.012 and +0.005 dex); 800 Angstrom (+0.022 and +0.019 dex)
  is close to iron/v4.0 (+0.024 and +0.020 dex).
- The median |`SMOOTHCORR`| is nearly independent of the knot spacing in
  every camera and sample, and smaller than in iron/v4.0 (e.g., 2.15% against
  3.07% in the r camera of the control). This is why `smooth-diagnostics` was
  written. The b camera of `calib` (median about 70%) is not informative.

### Where the broad lines are lost

Fraction of the gold BL-AGN with a broad line in iron/v4.0 but not in the arm,
against the observed wavelength of H-alpha (normalized by the number of gold
BL-AGN in each range):

| Observed H-alpha [Angstrom] | `knots200` | `knots400` | `knots800` | `nosmooth` |
|---|---|---|---|---|
| 6800 to 7400 (r camera) | 4.7 to 5.7% | 2.7 to 3.4% | 1.3 to 1.4% | 1.2 to 1.6% |
| 7400 to 7600 (r/z overlap) | 8.6% | 2.7% | 1.0% | 1.3% |
| 8000 to 9400 (z camera) | 1.6 to 2.9% | 1.1 to 2.1% | 0.3 to 1.4% | 0.2 to 1.0% |
| 9400 to 9600 (red end of z) | 5.0% | 4.7% | 2.7% | 0.3% |

- At 200 Angstrom there are two loss mechanisms. At z > 0.2 (H-alpha in the z
  camera) the losses are concentrated at FWHM > 5000 km/s (6 to 8%, against
  about 2% for narrower lines); at z = 0.1 to 0.2 the loss is 5 to 7% at every
  width. Of the 635 objects lost at 200 Angstrom, 399 have FWHM > 2000 km/s
  and 30 have FWHM < 1000 km/s.
- The red end of the z camera loses broad lines at every knot spacing.
- The cause has not been established; the per-camera spline with constant
  extrapolation at its ends is the obvious suspect when the line mask is near
  the end of a camera. This needs `fastqa` on a few of the lost objects.

### `smooth-diagnostics`: full run (2026-10-10)

`smooth-diagnostics.fits` holds all 40,897 objects and the five arms of the
branch (no `v4.0` arm). The numbers below were computed on the laptop directly
from that file and `emfit-sample.fits`, with the definitions of
`summarize_diag` in `compare-emfit`; `smoothdiag.txt` and `smoothdiag.png`
have not been made yet. `HA_BUMP` is NaN for 27 objects (0.07%; it was 15% in
the 20-healpix test), and one object has no r-camera chi2. The median chi2 per
pixel is 0.96 to 1.02 in every knot arm and camera of the control sample. The
median number of spline parameters in the b, r, and z cameras is 6, 5, and 6
at 800 Angstrom; 8, 8, and 9 at 400; 13, 12, and 14 at 200; and 22, 19, and 24
at 100.

Benefit: median decrease in chi2 per added spline parameter in the b / r / z
cameras (and, in parentheses, the percentage of objects in which the decrease
is significant at 3-sigma), relative to the next stiffer arm:

| Step | Control | `calib` | Gold BL-AGN |
|---|---|---|---|
| `nosmooth` to 800 | 2.5 / 2.1 / 3.3 (46 / 35 / 59) | 11.7 / 5.2 / 10.6 (84 / 70 / 88) | 21.5 / 6.2 / 8.6 (90 / 75 / 85) |
| 800 to 400 | 0.94 / 0.93 / 1.12 (8 / 7 / 11) | 1.15 / 0.87 / 1.17 (15 / 7 / 13) | 1.7 / 2.4 / 3.6 (28 / 37 / 51) |
| 400 to 200 | 1.04 / 0.94 / 1.16 (6 / 5 / 10) | 1.18 / 0.92 / 1.26 (12 / 7 / 14) | 2.0 / 1.9 / 2.2 (36 / 31 / 37) |
| 200 to 100 | 1.01 / 0.94 / 1.03 (3 / 3 / 4) | 1.07 / 0.91 / 1.04 (7 / 4 / 5) | 2.0 / 1.7 / 1.6 (42 / 31 / 32) |

- In the control sample the 800 Angstrom spline fits real structure (2 to 3
  per parameter), and every knot added beyond it fits noise (0.9 to 1.2). The
  same holds for `calib`, i.e., even where the old algorithm made its largest
  corrections there is nothing left for knots finer than 800 Angstrom.
- A minority of the control sample (7 to 11% from 800 to 400 Angstrom) does
  gain significantly; these objects have not been looked at.
- Only the BL-AGN gain at every step. The gain rises with the EmFit FWHM (z
  camera, 800 to 400 Angstrom, H-alpha in the z camera: 1.8, 2.4, and 5.2 for
  FWHM of 589 to 1500, 1500 to 3000, and above 3000 km/s), and it is not
  confined to the camera with H-alpha. This points to the AGN (its continuum,
  or broad features outside the line mask) and not to calibration, but we
  cannot separate it from the S/N, which also rises with the FWHM.

Cost, for the gold BL-AGN. The absorbed fraction is `HA_BUMP` over the EmFit
broad H-alpha flux, and losses and gains are relative to `nosmooth`:

| Arm | Absorbed fraction: 2nd / 50th / 98th percentile | Lost / gained | Lost with abs(fraction) > 0.1 | Loss rate where abs(fraction) < 0.02 | Control: NMAD of the bump EW [Angstrom] |
|---|---|---|---|---|---|
| `knots100` | -29% / 0.4% / 40% | 2258 / 204 | 1248 | 4.6% | 5.6 |
| `knots200` | -12% / 0.3% / 19% | 670 / 205 | 258 | 1.9% | 2.5 |
| `knots400` | -5% / 0.3% / 15% | 415 / 137 | 128 | 1.6% | 0.71 |
| `knots800` | -3% / 0.0% / 5% | 198 / 138 | 20 | 1.1% | 0.23 |

- The median absorbed fraction is zero in every arm; the cost is in the
  tails, and it is two-sided (a negative value means that the spline is
  higher at the ends of the window than under the line).
- The loss rate rises steeply with the absorbed fraction. At 200 Angstrom it
  is 1.9% for abs(fraction) < 0.02, 14% for 0.1 to 0.2, 28% for 0.2 to 0.4,
  and 71% above 0.4 (and 16% below -0.1). So the spline does remove broad
  lines; `FABS_LOST` against `FABS_KEPT` (0.027 against 0.003 at 200
  Angstrom) understates this, because most of the lost objects are not in
  the tails.
- Absorption is not the whole story. Of the 670 objects lost at 200 Angstrom,
  258 have abs(fraction) > 0.1, and about 300 are expected from the loss rate
  of the objects with no measurable bump. That rate itself rises with the
  flexibility (1.1, 1.6, 1.9, and 4.6% at 800, 400, 200, and 100 Angstrom),
  as does the scatter of the bump in the control sample, which has no broad
  line to absorb (0.23 to 5.6 Angstrom of equivalent width). The flexible
  splines add noise to the continuum under H-alpha in every object, which is
  consistent with the larger [NII] scatter.
- At 800 Angstrom the losses (198) and gains (138) nearly cancel, and only 20
  of the lost objects have abs(fraction) > 0.1.
- The camera ends remain a problem at every knot spacing. At 800 Angstrom,
  abs(fraction) > 0.1 in 7.9% of the gold BL-AGN with H-alpha at 9400 to 9600
  Angstrom and in 3.1% at 7200 to 7600 Angstrom, against 0.4% at 8000 to 9200
  Angstrom and 0.3% at 6800 to 7200 Angstrom; the loss rates are 4.0, 2.1,
  0.9, and 1.5%. This supports the suspicion above about the ends of the
  per-camera spline, and it is independent of the choice of knot spacing.

### 800 Angstrom with the current code (2026-10-11)

After the results above, the default became 800 Angstrom (fastspecfit
a451be6), a short run of unmasked pixels at either end of a camera is no
longer used if it is shorter than the masked region next to it (1fbef9b), and
the Monte Carlo realizations are fit at once with the pixels and knots of the
nominal fit (2b7acc7). The `knots800` arm was fit again with this code.
`compare-emfit` was run on 2026-10-11 with the eight arms, and
`smooth-diagnostics` with three (`knots800`, `old-knots800`, `nosmooth`), so
`smooth-diagnostics.fits` no longer has the 100, 200, and 400 Angstrom arms
of the tables above.

From `compare/summary.txt` and `compare/changed-<arm>.txt`:

| | `v4.0` | `old-knots800` | `knots800` | `nosmooth` |
|---|---|---|---|---|
| Gold recovered | 88.55% | 88.42% | 88.54% | 88.76% |
| FWHM < 1000 | 65.0% | 65.6% | 65.6% | 66.2% |
| 1000 to 2000 | 80.6% | 80.5% | 80.5% | 80.8% |
| FWHM > 2000 | 94.2% | 94.0% | 94.2% | 94.3% |
| log M_BH < 6 | 54.5% | 55.3% | 55.1% | 55.9% |
| Control with a broad line | 1.04% | 1.07% | 1.05% | 1.11% |
| FWHM offset / scatter vs EmFit [dex] | +0.024 / 0.030 | +0.022 / 0.030 | +0.022 / 0.030 | +0.036 / 0.034 |
| Broad flux offset / scatter vs EmFit [dex] | +0.020 / 0.031 | +0.019 / 0.033 | +0.020 / 0.033 | +0.031 / 0.034 |
| [NII] scatter of the control [dex] | 0.037 | 0.036 | 0.036 | 0.037 |
| Lost / gained / spurious vs `v4.0` | | 143 / 119 / 19 | 129 / 126 / 17 | 133 / 170 / 21 |

- `knots800` matches iron/v4.0 in every row, and its losses and gains
  relative to iron/v4.0 are at the floor set by `nosmooth`.
- `prebatch-knots800` has the same summary row as `knots800` to every digit,
  and `changed-prebatch-knots800.txt` is identical to `changed-knots800.txt`,
  so fitting the realizations with the rejected pixels of the nominal fit has
  no measurable effect.
- The benefit is unchanged (`compare/smoothdiag.txt`): relative to
  `nosmooth`, the median decrease in chi2 per spline parameter is 2.5 / 2.0 /
  3.4 in the b / r / z cameras of the control (significant in 46 / 35 / 59%),
  11.9 / 5.2 / 10.7 in `calib` (84 / 70 / 88%), and 21.9 / 5.9 / 8.6 in the
  BL-AGN (88 / 73 / 84%).

Median `SMOOTHRMS` (the rms of the smooth continuum relative to the median
stellar continuum, over all the pixels with data) of `knots800`, in percent,
from `compare/summary.txt`; see also `compare/smoothrms.png`:

| Sample | b | r | z |
|---|---|---|---|
| Control | 10.8 | 3.1 | 2.8 |
| Gold BL-AGN | 13.4 | 2.5 | 2.2 |
| `calib` | 30.3 | 7.1 | 7.7 |

- The smooth continuum is typically 2 to 3% of the stellar continuum in the
  r and z cameras and about 10% in the b camera, and two to three times
  larger in `calib`, as expected of a sample selected by the size of the old
  correction. In about 15% of `calib` it is at least 100% in the b camera.
- These numbers cannot be compared with the `SMOOTHCORR` of the other arms
  (e.g., a median of 3.8 / 2.1 / 1.7% in the control at 800 Angstrom before
  the change): `SMOOTHCORR` is a median of the signed ratio to the flux,
  which averages out a smooth continuum which changes sign across the
  camera, and `SMOOTHRMS` does not.

Cost, for the gold BL-AGN, relative to `nosmooth`. These numbers were
computed on the laptop from `smooth-diagnostics.fits` and `emfit-sample.fits`
(they are not in `compare-emfit`), for the objects fit in both arms with an
EmFit broad H-alpha flux:

| | `old-knots800` | `knots800` |
|---|---|---|
| Absorbed fraction: 2nd / 50th / 98th percentile | -3.3% / 0.0% / 5.4% | -2.1% / 0.0% / 3.4% |
| Lost / gained | 199 / 138 | 170 / 130 |
| Lost with abs(fraction) > 0.1 | 20 | 7 |
| Loss rate where abs(fraction) < 0.02 | 1.11% | 0.99% |
| Control: NMAD of the bump EW [Angstrom] | 0.23 | 0.19 |

Percentage of the gold BL-AGN with abs(fraction) > 0.1, and loss rate (lost /
gained), against the observed wavelength of H-alpha:

| Observed H-alpha [Angstrom] | N | `old-knots800` | `knots800` |
|---|---|---|---|
| 6800 to 7200 | 1,330 | 0.23%; 1.51% (17 / 19) | 0.00%; 1.51% (17 / 18) |
| 7200 to 7600 | 2,172 | 3.55%; 2.08% (37 / 20) | 1.43%; 1.63% (29 / 14) |
| 7600 to 8000 | 2,814 | 1.17%; 1.28% (32 / 23) | 0.50%; 1.24% (31 / 24) |
| 8000 to 9200 | 9,670 | 0.32%; 0.91% (80 / 59) | 0.21%; 0.72% (63 / 59) |
| 9200 to 9400 | 1,007 | 3.08%; 0.85% (8 / 7) | 0.50%; 0.64% (6 / 5) |
| 9400 to 9600 | 631 | 7.29%; 4.06% (23 / 3) | 2.54%; 4.06% (23 / 3) |

- Dropping the short runs at the ends of the cameras removes most of the
  absorption there (a factor of 2.5 to 6 fewer objects with abs(fraction) >
  0.1 in the three ranges near the ends), and with it about 30 of the lost
  broad lines.
- The red end of the z camera still loses broad lines: 23 lost and 3 gained
  with H-alpha at 9400 to 9600 Angstrom, the same as before the fix (21 of
  the 23 are the same objects). All 23 have H-alpha at 9400 to 9500 Angstrom
  (14 of 266 at 9400 to 9450, 9 of 216 at 9450 to 9500, none of 84 at 9500
  to 9550), and their EmFit FWHM ranges from 1419 to 8094 km/s (median 2925,
  against 2785 for all the objects in this range).
- Absorption as measured by `HA_BUMP` does not explain them: only 4 of the 23
  have abs(fraction) > 0.1 (median 0.008). `HA_BUMP` subtracts the straight
  line which joins the ends of the window, so it does not register an offset
  or a tilt of the smooth continuum under the line, e.g., from the constant
  extrapolation beyond the last pixel which is used. The cause is not known.
  This is about 20 objects, 0.1% of the gold sample.

### Where this leaves the default

The default was changed from 200 to 800 Angstrom on 2026-10-10
(`SMOOTH_KNOT_SPACING` in `py/fastspecfit/util.py`; fastspecfit a451be6).
Every measurement favors 800 Angstrom over 200 (the numbers below are those
of `old-knots800`):

- Recovery, scatter, and false positives at 800 Angstrom match iron/v4.0 and
  `nosmooth`; 400 Angstrom costs about 1.2 points of recovery and 200
  Angstrom 2.4.
- Knots finer than 800 Angstrom buy nothing in the control and `calib`
  samples, and they cost broad lines and narrow-line precision.
- 800 Angstrom is better than no smooth continuum: its knots fit real
  structure in 35 to 59% of the control sample (70 to 88% of `calib`), and
  the bias of the broad line relative to EmFit drops from +0.036 to +0.022
  dex in FWHM.

The one argument for 400 Angstrom is the BL-AGN themselves: their continuum
does gain from more knots, and the broad-line offsets relative to EmFit are
smallest there (+0.012 dex in FWHM). The default is meant for most objects,
so this does not outweigh the cost. No arm stiffer than 800 Angstrom was
run, so we do not know where between 800 Angstrom and no smooth continuum the
benefit falls off.

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

## Status and next steps

Status on 2026-10-11:

- All the arms of the table above are fit, and `compare-emfit` and
  `smooth-diagnostics` (three arms) were run at NERSC on 2026-10-11. Their
  output, `emfit-sample.fits`, and `smooth-diagnostics.fits` are also on the
  laptop; the per-arm catalogs and per-healpix files are only at NERSC
  (`$PSCRATCH/fastspecfit/emfit`, which is purged).
- `compare-emfit` now reads `SMOOTHRMS` as well as `SMOOTHCORR` (a column
  which is not in a catalog is NaN and not zero) and writes `smoothrms.png`.
  It was run again with this change, and `compare/` on the laptop is its
  output.
- The branch was assessed on 2026-10-11 as ready for a PR, once the two
  items of step 1 are fixed. The tests were not run as part of that
  assessment.

Next steps, in order:

1. In the fastspecfit repo, before the PR: `etc/fastspecfit-env.sh` (acccd49)
   has `module load fastspecfit` commented out and the development checkout
   in `PYTHONPATH` and `PATH`; and `doc/changes.rst` still advertises
   `--smooth-window` and `--smooth-step`, and has no entry for the new
   algorithm, `--smooth-knot-spacing`, the `SMKNOTS` header keyword, or the
   columns which replace `SMOOTHCORR_*`.
2. `fastqa` on the lost objects with H-alpha at 9400 to 9500 Angstrom, to
   find out why they are lost, or state it in the PR as a known limitation.
3. Optional: a `knots1600` arm, to show that 800 Angstrom is not still more
   flexible than it needs to be.
4. Open a PR in fastspecfit (`smooth-cont` into `main`). Replacing the
   smooth-continuum algorithm is a fairly major change, so the PR should show
   the performance against EmFit, including the new diagnostics.

Open items:

- `fastqa` on a subset of `calib` at 200 and 800 Angstrom, and on a few of the
  control objects which gain significantly from 800 to 400 Angstrom.
- The loss rate against the absorbed fraction, and the absorbed fraction
  against the observed wavelength of H-alpha, are not in `compare-emfit`
  (computed by hand; see above).
- The plot colors were taken from a validated palette but not re-validated
  (no `node` on the laptop).
- Possible additions: recovery in bins of observed H-alpha wavelength (now
  computed by hand; see above); comparison of the fastspecfit smooth continuum
  at H-alpha with EmFit's `NII_HA_CONTINUUM`; H-beta broad-line comparison
  (the columns are already in both files).
