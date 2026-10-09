#!/bin/bash -l
#SBATCH --account=desi
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --mail-user=jmoustakas@siena.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --time=04:00:00
#SBATCH --job-name=fastspec-emfit
#SBATCH --output=fastspec-emfit-%j.log

# Fit the EmFit comparison sample (see build-emfit-sample) with the
# smooth-cont branch of fastspecfit, once per "arm". Each arm is fit and then
# merged into ${rundir}/${arm}/iron/catalogs/fastspec-iron-emfit.fits.
#
# Existing per-healpix output files are skipped, so the job can be resubmitted
# if it times out.

# ---------------------------------------------------------------------------
# Edit the variables below before submitting.
# ---------------------------------------------------------------------------
N=4                     # number of nodes (must match --nodes above)
mp=16                   # multiprocessing workers per MPI rank
specprod=iron
fastspecfit_dir=/global/common/software/desi/users/ioannis/fastspecfit   # checkout of the smooth-cont branch
rundir=${PSCRATCH}/fastspecfit/emfit
samplefile=${rundir}/emfit-sample.fits
# knotsNNN = --smooth-knot-spacing=NNN [Angstrom]; nosmooth = --no-smooth-continuum
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

ntasks=$(( 128 * N / mp ))
cpus_per_task=$(( mp * 2 ))

for arm in ${arms}; do
    case "${arm}" in
        nosmooth) armargs="--no-smooth-continuum" ;;
        knots*)   armargs="--smooth-knot-spacing=${arm#knots}" ;;
        *)        echo "Unrecognized arm ${arm}"; exit 1 ;;
    esac

    args="--specprod=${specprod} --coadd-type=healpix --samplefile=${samplefile}"
    args+=" --outdir-data=${rundir}/${arm}"

    cmd="srun --nodes=${N} --ntasks=${ntasks} --cpus-per-task=${cpus_per_task} --cpu-bind=none"
    cmd+=" ${mpiscript} ${args} --mp=${mp} ${armargs}"
    echo "${cmd}"
    time ${cmd}

    cmd="srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=none"
    cmd+=" ${mpiscript} ${args} --mp=32 --nompi --merge --merge-suffix=emfit --overwrite"
    echo "${cmd}"
    time ${cmd}
done
