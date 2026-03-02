# AMREX_HOME defines the directory in which we will find all the AMReX code
AMREX_HOME ?= ../../../../amrex

# TOP defines the directory in which we will find Source, Exec, etc
TOP = ../../..

# compilation options
COMP    = gcc
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_FFT = TRUE
NO_MPI_CHECKING=FALSE

PRECISION = DOUBLE
DEBUG     = FALSE 

TINY_PROFILE = FALSE

# physics
DIM      = 3
# USE_GRAV = FALSE
# USE_FDM = FALSE
# USE_AXCOMPLEX = FALSE
# USE_FDM_SLIM_STATE = FALSE
# NO_HYDRO = TRUE
USE_INFLATION = TRUE

Bpack := ./Make.package
Blocs := .

LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3_omp -lfftw3
INCLUDE_LOCATIONS       += $(FFTW_DIR)/../include

include $(TOP)/Exec/Make.Nyx
