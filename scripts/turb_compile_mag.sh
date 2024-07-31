# go to Athena directory
source /u/wuze/.bashrc
echo $ATHENA_DIR
cd $ATHENA_DIR

# configure the makefile
./configure.py \
--prob turb_v2 \
-b  \
-fft -mpi  \
-hdf5 -h5double  \
--mpiccmd mpiicc  \
--nscalars=1  \
--include=$FFTW_HOME/include  \
--include=$HDF5_HOME/include  \
--lib_path=$FFTW_HOME/lib  \
--lib_path=$HDF5_HOME/lib  \
--cflag="-xCORE-AVX512  \
-qopt-zmm-usage=high  \
-inline-forceinline  \
-qopenmp-simd   \
-qopt-prefetch=4  \
-qoverride-limits  \
-diag-disable 3180  \
-Wl,-rpath=$FFTW_HOME/lib  \
-Wl,-rpath=$HDF5_HOME/lib"

# make the executable
make clean
make -j5