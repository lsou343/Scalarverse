# AMREX_HOME defines the directory in which we will find all the AMReX code
AMREX_HOME ?= ../../../subprojects/amrex

# TOP defines the directory in which we will find Source, Exec, etc
TOP = ../../..

# compilation options
COMP    = gcc
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
NO_MPI_CHECKING=TRUE
USE_PARTICLES=FALSE
PRECISION = DOUBLE
DEBUG     = FALSE

# physics
DIM      = 3
USE_GRAV = FALSE
USE_FDM = FALSE
NO_HYDRO = TRUE
USE_AXCOMPLEX = TRUE

Bpack := ./Make.package
Blocs := .

LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3_omp -lfftw3
INCLUDE_LOCATIONS       += $(FFTW_DIR)/../include

include $(AMREX_HOME)/Src/Extern/SWFFT/Make.package
INCLUDE_LOCATIONS       += $(AMREX_HOME)/Src/Extern/SWFFT
VPATH_LOCATIONS         += $(AMREX_HOME)/Src/Extern/SWFFT

include $(TOP)/Exec/Make.Nyx
