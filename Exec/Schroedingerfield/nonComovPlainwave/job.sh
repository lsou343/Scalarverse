#!/bin/bash 

export FFTW_DIR='/opt/sw/rev/23.12/linux-scientific7-cascadelake/gcc-9.5.0/fftw-3.3.8-62yh7t/lib'
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH

module load openmpi/4.1.6 gcc/9.5.0 fftw/3.3.10 hdf5/1.14.3 python/3.9.0


 # mpirun ./AxNyx3d.gnu.DEBUG.MPI.ex inputs 
# run gdb instead
gdb --args mpirun ./AxNyx3d.gnu.DEBUG.MPI.ex inputs
