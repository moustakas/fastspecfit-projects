# Handoff: tuning the fastspecfit smooth continuum against EmFit (DR1)

Written 2026-10-09. No code exists yet; this folder holds only this file.
Start here, then delete or fold this into a `README.md` once the scripts exist.

## Goal

Tune the smooth-continuum parameters on the fastspecfit `smooth-cont` branch
using galaxies with broad (but not very broad) Balmer lines, i.e., low-mass
black-hole hosts, with the EmFit DR1 VAC (Pucha+25, Pucha+26) as the external
reference. John's framing: "This test is just to try to tune the
smooth-continuum parameters." Keep it simple.

Exact agreement with EmFit is not expected: EmFit shifts to the rest frame and
resamples, subtracts a FastSpecFit stellar continuum (from the public DR1
FastSpecFit VAC, not from v4.0), and fits multiple kinematic components.

## Where things are

| What | Where |
|---|---|
| fastspecfit repo, branch `smooth-cont` | `~/code/desihub/fastspecfit` (HEAD `09a52d5` on 2026-10-08) |
| This project (pipeline + analysis scripts) | `~/code/fastspecfit-projects/emfit/` |
| EmFit VAC docs | https://data.desi.lbl.gov/doc/releases/dr1/vac/emfit |
| EmFit catalog at NERSC | `/global/cfs/cdirs/desi/public/dr1/vac/dr1/emfit/` (`emfit-dr1-v2.3.1.fits`, 17 GB, one HDU `EMFIT`, 506 columns, 7,378,347 rows) |
| EmFit catalog locally | Still downloading as of 2026-10-09; ask John for the path |
| Baseline fastspec results | `/dvs_ro/cfs/cdirs/desi/vac/dr1/fastspecfit/iron/v4.0/catalogs/` (tag 3.6.1) |
| Pucha+26 PDF | `~/research/bibdesk/pucha/26pucha_a a new record census of dwarf agn and a bimodal m$_bh$ - m$_\star$ scaling relation with desi dr1.pdf` (Appendix A is pp. 19 to 26, Appendix B pp. 26 to 27) |
| Pucha+26 supplementary catalogs (AGN candidates, full BL-AGN sample) | https://doi.org/10.5281/zenodo.21539891 |
| Samplefile workflow | `etc/README.sample`, `etc/fastspecfit-sample.slurm`, `etc/fastspecfit-sample.sh`, `etc/fastspecfit-env.sh` in the fastspecfit repo |

## What the branch changes

- The sliding-window median smooth continuum is replaced by a per-camera
  fixed-knot spline fit to the unmasked pixels
  (`ContinuumTools.smooth_continuum` in `py/fastspecfit/continuum.py`).
- Tunables: `--smooth-knot-spacing` (default `SMOOTH_KNOT_SPACING = 200` Angstrom
  in `util.py`) and, inside the function, the minimum number of pixels between
  knots (default 15) used to prune knots across masked regions.
- The smooth continuum is now Monte Carloed and passed to `emline_specfit` as
  `smooth_continuum_monte`.
- `--no-smooth-continuum` still exists; `--tauv-bounds` is new.
- Header cards `SMKNOTS` and `TAUVBND` record the settings.

Why broad lines are the stress test: a 200 Angstrom spline cannot by itself
absorb a broad H-alpha of FWHM 500 to 2000 km/s (about 10 to 45 Angstrom), so
the risk is in (a) broad wings leaking outside the line mask into the spline
fit, (b) knot pruning across a wide masked H-alpha+[NII] region, and (c) the
downstream broad-versus-narrow model selection (`minsnr_balmer_broad`,
`minsigma_balmer_broad`) reacting to the changed residual.

## Decisions John has made

1. **Baseline is the iron/v4.0 VAC**, not a rerun of `main`. It was run with
   tag 3.6.1 and "has all the same defaults except the smooth-continuum MC".
   So only the branch arms need to be run.
2. **Sample matches the Pucha+ cuts**; keep it as simple as possible.
3. **No mini-specprod.** The sample is small enough to just run everything at
   NERSC.
4. Scripts live here, not in the fastspecfit repo. Runs use `mpi-fastspecfit
   --samplefile` at NERSC.

Standing preferences that matter here (from John's global memory): he runs
scripts, tests, and installs himself, so draft and `py_compile` only; never
commit or push without approval; ask rather than search when a path is unknown.

## What Pucha+26 says (reviewed 2026-10-09)

### Data model (Appendix A.1, Tables 2 and 3)

- Identifiers: `TARGETID`, `SPECPROD` (`iron` for DR1), `SURVEY`, `PROGRAM`,
  `HEALPIX`, `Z`, plus `TARGET_RA`, `TARGET_DEC` in v2.3.1. These are exactly
  the samplefile columns `mpi-fastspecfit` needs.
- `Z` is the Redrock redshift after QuasarNet corrections, which is also the
  fastspec default, so `--input-redshifts` should not be needed.
- `PROB_BROAD`: percentage of iterations in which a broad H-alpha is detected.
- Per component (`{EMLINE}_`): `AMPLITUDE`, `MEAN`, `STD`, `FLUX`, `SIGMA`
  with `_ERR` (and `_LERR`, `_UERR` for flux and sigma), and `SIGMA_FLAG`.
  Fluxes are in 1e-17 erg/s/cm2; `SIGMA` is in km/s and corrected for
  instrumental resolution using the median resolution element in the window.
  `SIGMA_FLAG`: 0 resolved and corrected, 1 unresolved, -1 not detected (all
  columns zero).
- Per window (`{WINDOW}_`): `CONTINUUM`, `CONTINUUM_ERR`, `NOISE`, `NDOF`,
  `RCHI2`.
- Windows and components:
  - default mode: `HB` (`HB_N`, `HB_OUT`, broad H-beta), `OIII` (`OIII4959`,
    `OIII5007`, each with `_OUT`), `NII_HA` (`NII6548`, `NII6583`, `HA_N`,
    `HA_OUT`, `HA_B`, plus `_OUT` for [NII]), `SII` (`SII6716`, `SII6731`,
    each with `_OUT`)
  - EBL mode: `HB_OIII` and `NII_HA_SII`, same components
  - Table 3 prints `HB_OUT` twice; the broad H-beta is presumably `HB_B`.
    Confirm from the file.
- Mode is inferred from the windows: in default mode `HB_OIII_NDOF`,
  `HB_OIII_RCHI2`, `NII_HA_SII_NDOF`, `NII_HA_SII_RCHI2` are zero. Counts:
  7,363,400 default, 14,947 EBL.
- `OIII_DBL_FLAG`, `SII_DBL_FLAG`: double narrow peaks; when True, sum
  `*_FLUX` and `*_OUT_FLUX` (errors in quadrature). `SII_DBL_FLAG` also governs
  [NII], H-alpha, H-beta. Counts: 18,446 and 28,293.
- EmFit fits a low-order polynomial for residual continuum in each window
  (`*_CONTINUUM`). This is the closest analog of our smooth continuum and is
  worth plotting against it.
- All wavelengths are vacuum.

### Selection (Section 2.5), in order

1. `ZCAT_PRIMARY`, `COADD_FIBERSTATUS == 0`, `ZWARN` in (0, 4), `SPECTYPE`
   `GALAXY` or `QSO`, 0.001 <= z <= 0.45: 7,434,906.
2. LS DR9: SNR >= 5 and `FRACFLUX` <= 0.25 in g, r, z: 6,856,046.
3. CIGALE (DR1 physical-properties VAC): chi2 <= 10, log M* >= 6, error on
   log M* <= 0.5 dex, 0.2 <= `FLAG_MASSPDF` <= 5: 6,143,599 (6,099,572 with
   EmFit measurements).
4. Line-emitting: SNR >= 3 for [OIII], H-alpha, [NII]; SNR >= 1 and AoN >= 1
   for H-beta (after summing double-peaked components): 1,678,787.
5. Broad-line candidates: `PROB_BROAD` >= 80, SNR(H-alpha broad flux) >= 3,
   SNR(broad sigma) >= 3, AoN(H-alpha broad) >= 2: **26,588** (3,246 dwarf with
   log M* <= 9.5, 23,342 high-mass).
6. BL-AGN: of those, [NII]-BPT AGN or composite (narrow components only):
   19,432.
7. Confident BL-AGN after stacking and visual inspection of FWHM < 1000 km/s
   candidates (Appendix B): **17,949** (13,026 regular, 4,923 EBL; 15,871
   extended, the rest `TYPE == PSF`). Of these, 792 have log M_BH < 6.

Steps 2 and 3 need LS DR9 Tractor quantities and the CIGALE VAC; steps 4 to 6
need only EmFit columns; step 7 exists only in the Zenodo catalog.

### Caveats that bear on this test (Appendix A.3, B)

- Broad components are accepted at FWHM >= 300 km/s (sigma of about 127
  km/s), and Pucha urges caution below 1000 km/s. In the stacking analysis of
  the 2,099 extended candidates with FWHM < 1000 km/s, 1,159 were removed
  outright and 221 of 299 "tentative" ones were missed outflows. Only 226
  low-FWHM candidates survive. So raw EmFit broad components below 1000 km/s
  are not ground truth; the confident list is.
- A single Gaussian is used for the broad component, so complex profiles are
  approximated.
- Regular BL-AGN have median FWHM of about 1970 km/s; EBL have about 4700
  km/s. The low-mass regime John cares about is the regular (default-mode)
  population, mostly below about 2000 km/s.
- M_BH (their Eq. 2, Reines+13): log M_BH = 6.57 + 0.47 log(L_Ha,b / 1e42) +
  2.06 log(FWHM_Ha,b / 1e3 km/s), with epsilon = 1; cosmology is Planck 2020
  (H0 = 67.4, Omega_m = 0.315).

## Plan

### Scripts to write here (extensionless executables, as elsewhere in this repo)

| File | Purpose |
|---|---|
| `build-emfit-sample` | Read only the needed EmFit columns with `fitsio`, apply the cuts, write (a) the samplefile (`SURVEY`, `PROGRAM`, `HEALPIX`, `TARGETID`) and (b) a slim reference catalog with the broad and narrow H-alpha, H-beta, [NII], [SII], [OIII] columns, the window continuum, noise, and chi2 columns, the flags, and a derived mode flag. Runs at NERSC against the on-disk catalog. |
| `fastspec-emfit.slurm` | Adapted from `etc/fastspecfit-sample.slurm`; one `--outdir-data` per arm; merge step per arm. |
| `compare-emfit` | Row-match each arm and the v4.0 baseline to the slim catalog; compute the metrics; make figures; write outlier TARGETID lists for `fastqa`. |
| `README.md` | The recipe end to end. |

### Arms

- Baseline: iron/v4.0 (read from the VAC, no run needed).
- Branch at `--smooth-knot-spacing` 200 (default) plus a small grid (proposed
  100, 400; see open questions).
- Branch with `--no-smooth-continuum` as the "no correction" limit.

Cost: about 8 s/object/core for bright targets (`etc/README.sample`), so about
20k to 27k objects is under one node-hour per arm on Perlmutter.

### Metrics

- Recovery: fraction of EmFit broad H-alpha sources for which fastspec selects
  the broad model, versus EmFit broad FWHM, broad flux SNR, and broad-to-narrow
  flux ratio.
- Broad H-alpha (and H-beta) flux and sigma residuals against EmFit.
- Narrow H-alpha, [NII], [SII], [OIII], H-beta residuals; narrow H-alpha and
  [NII] are where a mis-subtracted broad pedestal shows up.
- Our smooth continuum at H-alpha versus EmFit's `NII_HA_CONTINUUM`.
- Everything split by default versus EBL mode, and with outflow
  (`*_OUT`, non-double-peaked) objects flagged, since fastspec has no outflow
  component.

## Open questions for John

1. **Which realization of "the Pucha+ cuts"?** Options:
   (a) the Zenodo confident BL-AGN list (17,949), which already encodes the
   visual vetting of the low-FWHM regime; (b) EmFit-only cuts (steps 4 and 5,
   optionally 6), which need no other catalog but give a superset of the 26,588
   because the photometry and CIGALE cuts are skipped; (c) the full recreation
   with the CIGALE VAC and Tractor photometry. Recommendation: (a) if the
   Zenodo file carries `TARGETID`, joined to EmFit for the measurements;
   otherwise (b).
2. **Low-mass black-hole subset.** Is the whole BL sample the tuning set, with
   the low-mass regime as a slice (FWHM < 1000 or < 2000 km/s, or log M_BH <
   6), or should the sample itself be restricted? Recommendation: fit the
   whole list and slice in the analysis.
3. **Knot-spacing grid.** Which values beyond 200 Angstrom, and should the
   minimum-pixels-between-knots parameter (currently not on the command line)
   also be varied?
4. **Figure of merit.** What decides the "tuned" value: broad-line recovery,
   broad flux agreement, narrow-line agreement, or visual QA of outliers?
5. **Narrow-line control sample.** Tuning on broad-line objects alone cannot
   see a setting that creates spurious broad lines. Add a matched sample of
   EmFit narrow-line objects (similar size), or skip for simplicity?
6. **Output location at NERSC.** Default assumed:
   `$PSCRATCH/fastspecfit/emfit/<arm>/`.

## Things to verify before writing code

- **`minsigma_balmer_broad`**: `emline_specfit` defaults to 250 km/s (FWHM of
  about 590 km/s), while EmFit's floor is FWHM 300 km/s. If the 250 km/s
  threshold rejects narrower broad components, EmFit candidates below about
  590 km/s are unrecoverable by construction, independent of the smooth
  continuum. Read how the threshold is applied in `emlines.py` and exclude or
  separately bin that regime.
- **Baseline confounds**: v4.0 (3.6.1) and the branch differ by more than the
  smooth continuum: PR #289 (H6 to H8 rename, [SIII] wavelengths, derived
  doublet-ratio columns) and different Monte Carlo seeds. These should not
  matter near H-alpha, but confirm the v4.0 column names being compared.
- **EmFit column names**: confirm `HA_B_FLUX`, `HA_B_SIGMA`, `HA_B_AMPLITUDE`,
  `NII_HA_NOISE`, `PROB_BROAD`, and the broad H-beta name directly from the
  file. AoN is presumably amplitude over window noise; SIGMA to FWHM is 2.355.
- **v4.0 layout**: list the catalogs directory to see the file naming (merged
  per survey/program, or per healpix) and whether the `MODELS` spectra are
  available for QA.
- **Zenodo catalog**: check its columns (`TARGETID`? the confident flag?
  M_BH?).

John has the catalog at NERSC and offered to confirm any of this. The quickest
checks for him to run there:

```bash
ls /global/cfs/cdirs/desi/public/dr1/vac/dr1/emfit/
ls /dvs_ro/cfs/cdirs/desi/vac/dr1/fastspecfit/iron/v4.0/catalogs/ | head -30
python -c "
import fitsio
F = fitsio.FITS('/global/cfs/cdirs/desi/public/dr1/vac/dr1/emfit/v2.3/emfit-dr1-v2.3.1.fits')
print(F[1].get_nrows()); print('\n'.join(F[1].get_colnames()))
"
```

The `v2.3/` subdirectory in that last path is a guess from the docs ("maintained
within the v2.3 directory structure"); adjust to whatever the first `ls` shows.

## Side finding (not acted on)

`doc/changes.rst` on the branch still advertises `--smooth-window` and
`--smooth-step` under PR #292, but the command line now exposes
`--smooth-knot-spacing` (PR #291). The change log entry is stale.
