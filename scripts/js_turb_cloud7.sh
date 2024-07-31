#!/usr/bin/env bash
# source ~/.bashrc

#SBATCH -J 0613_0.3_3_cloud
#SBATCH -o ./out/240613/0.3_3_cloud."%j".out
#SBATCH -e ./out/240613/0.3_3_cloud."%j".err
#SBATCH --mail-user wuz@mpa-garching.mpg.de
#-- wuz@mpa-garching.mpg.de
#SBATCH --partition=p.24h
#-- p.24h, p.test, p.gpu & p.gpu.ampere
#SBATCH --mail-type=ALL
#SBATCH --nodes=2
#-- 2 at most
#SBATCH --ntasks-per-node=32
#-- 40 at most
#SBATCH --time=04:00:00
#-- in format hh:mm:ss

set -e
SECONDS=0

module purge
module load intel/19.1.2
module load impi/2019.8
module load fftw-mpi/3.3.8
module load hdf5-mpi/1.8.21
module load ffmpeg/4.4
module list

mkdir -p /freya/ptmp/mpa/wuze/out/240613
cd $DP/240612_0.8_12/cloud

srun ../../athena_turb -i athinput_cloud.turb -t 03:55:00
#-- !!TIME_LIMIT_RST!! should be ten minutes before the end, to generate a restart file

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"

cd ~