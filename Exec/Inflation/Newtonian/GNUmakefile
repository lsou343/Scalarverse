# AMREX_HOME defines the directory in which we will find all the AMReX code
AMREX_HOME ?= ../../../../amrex

# TOP defines the directory in which we will find Source, Exec, etc
TOP = ../../..

# compilation options
COMP    = gcc
USE_MPI = TRUE
USE_FFT = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
NO_MPI_CHECKING = FALSE

# USE_AMREX_PARTICLES = TRUE
PRECISION = DOUBLE
DEBUG     = FALSE

# physics
DIM      = 3
USE_GRAV = TRUE
# NO_HYDRO = TRUE (need to remove)
USE_FDM = FALSE
USE_INFLATION = TRUE
USE_COMOV_FULL = TRUE
USE_COMOV_EOS = FALSE
USE_SCHROEDINGER = FALSE
USE_NEWT = TRUE

TINY_PROFILE = FALSE

Bpack := ./Make.package
Blocs := .

# Libraries and include locations
LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3_omp -lfftw3
INCLUDE_LOCATIONS       += $(FFTW_DIR)/../include

# For some reason particle folder is not loaded by default
INCLUDE_LOCATIONS       += $(AMREX_HOME)/Src/Particle

# Include the path to MLMG headers
INCLUDE_LOCATIONS       += $(AMREX_HOME)/Src/LinearSolvers/MLMG

include $(TOP)/Exec/Make.Nyx

# compile command
# export FFTW_DIR=‘/opt/sw/rev/23.12/linux-scientific7-cascadelake/gcc-9.5.0/fftw-3.3.8-62yh7t/lib’
# export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH
# module load openmpi/4.1.6 gcc/9.5.0 fftw/3.3.10 hdf5/1.14.3 python/3.9.0
# make -j20
