#!/bin/sh
#
#SBATCH -J tracer_avg
#SBATCH -o /ptmp/mpa/wuze/multiphase_turb/out/240807/tracer_avg."%j".out
#SBATCH -e /ptmp/mpa/wuze/multiphase_turb/out/240807/tracer_avg."%j".err
#SBATCH --mail-user wuz@mpa-garching.mpg.de
#SBATCH --partition=p.test
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00

set -e
SECONDS=0

module purge
module list

cd /ptmp/mpa/wuze/multiphase_turb/codes/processing

# set number of OMP threads *per process*
export OMP_NUM_THREADS=1

srun python tracer_avg_evol_run.py '240716_0.6_6.4' $SLURM_CPUS_PER_TASK

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"