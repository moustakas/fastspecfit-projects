#!/bin/bash -l
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=5
#SBATCH --time=02:00:00
#SBATCH --job-name=fastspec-emfit
#SBATCH --output=fastspec-emfit-%j.log

# Fit the EmFit comparison sample (see build-emfit-sample) with the
# smooth-cont branch of fastspecfit, once per "arm". Each arm is fit and then
# merged into ${rundir}/${arm}/iron/catalogs/fastspec-iron-emfit.fits.
#
# The arms run concurrently, one node each, without MPI: a single process per
# node with one multiprocessing worker per physical core, each fitting one
# healpix file at a time. (With MPI, job 59592459 launched 32 independent
# "rank 0" processes which all fit the full sample.)
#
# Existing per-healpix output files are skipped, so the job can be resubmitted
# if it times out.

# ---------------------------------------------------------------------------
# Edit the variables below before submitting.
# ---------------------------------------------------------------------------
mp=128                  # multiprocessing workers per node
specprod=iron
fastspecfit_dir=/global/common/software/desi/users/ioannis/fastspecfit   # checkout of the smooth-cont branch
rundir=${PSCRATCH}/fastspecfit/emfit
samplefile=${rundir}/emfit-sample.fits
# knotsNNN = --smooth-knot-spacing=NNN [Angstrom]; nosmooth = --no-smooth-continuum
# The number of arms must match --nodes above.
arms="knots100 knots200 knots400 knots800 nosmooth"
# ---------------------------------------------------------------------------

# Use the development checkout in place of the fastspecfit module, with its
# own Numba cache.
export NUMBA_CACHE_DIR=${PSCRATCH}/fastspecfit/numba-cache/smooth-cont
source "${fastspecfit_dir}/etc/fastspecfit-env.sh"
export PYTHONPATH=${fastspecfit_dir}/py:${PYTHONPATH}
export PATH=${fastspecfit_dir}/bin:${PATH}

mpiscript=$(type -p mpi-fastspecfit)
echo "Using ${mpiscript}"
python -c "import fastspecfit; print(fastspecfit.__file__)"

# Warm up the Numba cache on a single rank (see etc/fastspecfit-sample.slurm).
echo "Warming up Numba cache on one rank..."
read wu_survey wu_program wu_healpix <<< $(python -c "
from astropy.table import Table; t = Table.read('${samplefile}')[0]
print(t['SURVEY'], t['PROGRAM'], t['HEALPIX'])")
srun --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=cores \
    $mpiscript \
    --specprod=${specprod} --coadd-type=healpix \
    --survey=${wu_survey} --program=${wu_program} --healpix=${wu_healpix} \
    --mp=1 --ntargets=1 --nmonte=0 --nompi \
    --outdir-data=/tmp/fastspecfit-warmup --overwrite
echo "Warm-up complete."

run_arm() {
    arm=$1
    case "${arm}" in
        nosmooth) armargs="--no-smooth-continuum" ;;
        knots*)   armargs="--smooth-knot-spacing=${arm#knots}" ;;
        *)        echo "Unrecognized arm ${arm}"; return 1 ;;
    esac

    args="--specprod=${specprod} --coadd-type=healpix --samplefile=${samplefile}"
    args+=" --outdir-data=${rundir}/${arm} --nompi"
    srun="srun --nodes=1 --ntasks=1 --cpus-per-task=256 --cpu-bind=none"

    cmd="${srun} ${mpiscript} ${args} --mp=${mp} ${armargs}"
    echo "${cmd}"
    ${cmd} > ${rundir}/fastspec-emfit-${arm}-${SLURM_JOB_ID}.log 2>&1

    cmd="${srun} ${mpiscript} ${args} --mp=32 --merge --merge-suffix=emfit --overwrite"
    echo "${cmd}"
    ${cmd} > ${rundir}/merge-emfit-${arm}-${SLURM_JOB_ID}.log 2>&1
    echo "Finished ${arm} at $(date)"
}

for arm in ${arms}; do
    run_arm ${arm} &
done
wait
