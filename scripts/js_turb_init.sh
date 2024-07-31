source ~/.zshrc

#SBATCH -J !!JOB_NAME!!
#SBATCH -o !!JOB_NAME!!."%j".out
#SBATCH -e !!JOB_NAME!!."%j".err
#SBATCH --mail-user !!USER_EMAIL!!
#-- wuz@mpa-garching.mpg.de
#SBATCH --partition=!!QUEUE!!
#-- p.24h, p.test, p.gpu & p.gpu.ampere
#SBATCH --mail-type=ALL
#SBATCH --nodes=!!NODES!!
#-- 2 at most
#SBATCH --ntasks-per-node=!!NTASKS_PER_NODE!!
#-- 40 at most
#SBATCH --time=!!TIME_LIMIT!!
#-- in format hh:mm:ss

set -e
SECONDS=0

module purge
module load intel/19.1.2
module load impi/2019.8
module load fftw-mpi/3.3.8
module load hdf5-mpi/1.8.21
module list

cd $DP/!!WORKING_DIR!!

srun ../athena_turb -i !!INPUT_FILE!! -t !!TIME_LIMIT_RST!!
#-- !!TIME_LIMIT_RST!! should be ten minutes before the end, to generate a restart file

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"