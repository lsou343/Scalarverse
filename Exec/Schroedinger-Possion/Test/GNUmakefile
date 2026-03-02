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
NO_MPI_CHECKING= FALSE

PRECISION = DOUBLE
# DEBUG     = TRUE 
DEBUG	 = FALSE
TINY_PROFILE = FLASE



# USE_AMREX_PARTICLES = TRUE
# For some reason particle folder is not loaded by default
# INCLUDE_LOCATIONS       += $(AMREX_HOME)/Src/Particle


# physics
DIM      = 3
USE_GRAV = TRUE
# USE_FDM = TRUE
# USE_AXCOMPLEX = FALSE
# USE_FDM_SLIM_STATE = FALSE
# NO_HYDRO = TRUE
# USE_INFLATION = TRUE
#USE_COMOV_FULL = TRUE
USE_COMOV_EOS = TRUE
USE_SCHROEDINGER = TRUE

Bpack := ./Make.package
Blocs := .

# FFTW_DIR ?= $(HOME)/local
# LIBRARIES += -L$(FFTW_DIR)/lib -lfftw3_mpi -lfftw3_omp -lfftw3
# INCLUDE_LOCATIONS       += $(FFTW_DIR)/include
LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3_omp -lfftw3
INCLUDE_LOCATIONS       += $(FFTW_DIR)/../include

#include $(AMREX_HOME)/Src/Extern/SWFFT/Make.package
#INCLUDE_LOCATIONS       += $(AMREX_HOME)/Src/Extern/SWFFT
#VPATH_LOCATIONS         += $(AMREX_HOME)/Src/Extern/SWFFT

include $(TOP)/Exec/Make.Nyx
