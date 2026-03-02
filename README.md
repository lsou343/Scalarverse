# Scalarverse

***NOTE: This README is a work in progress.***

## About

**Scalarverse** is a fork/re-write of **Axionyx** which is a fork of **Nyx**. 

- **Nyx** is a cosmological hydrodynamics/n-body solver that implements the AMReX adaptive mesh toolkit to focus computational resources where they're needed most. 
  - **Axionyx** implements the Schrödinger-Poisson equation into Nyx
    - **Scalarverse** implements a Klein-Gordon solver and a consistent scale-factor solver a la LatticeEasy (the Nyx/Axionyx scale-factor evolution is computed based on a fixed equation of state, which is not consistent with inflation, the primary use-case of the KG equation solver).

**Scalarverse** is kept compatible with the latest release of **AMReX**. Please download **AMReX** from <https://github.com/AMReX-Codes/amrex>, and switch to the latest release with

    git remote update
    tag=$(git describe --tags `git rev-list --tags --max-count=1`)
    git checkout tags/$tag


## Installation/Running

## Compile Flags 

**Scalarverse** is filled to the brim with both compile flags and runtime input parameters. These can each be divided into three categories, flags/parameters specific to Amrex, Nyx, and Scalarverse. The tables below are a (not yet, but will be) comprehensive list of compile flags and runtime input parameters.

### Scalarverse Compile Flags

| Flag | Description | Conflicts | Default | Suggested Value |
| --- | --- | --- | --- | --- |
| USE_INFLATION | Turns on/off the KG solver. | [Axionyx, AxReal, AxComplex] | False | None |

### Nyx Compile Flags

***None currently in use.***

### AMReX Compile Flags

| Flag | Description | Conflicts | Default | Suggested Value |
| --- | --- | --- | --- | --- |
| COMP | The C++ compiler to use. Ex. gcc (the best), LLVM, etc. | None. | None. | gcc |
| USE_MPI | Whether or not to compile with MPI multithreading. This is a good one, should be used. | None. | None. | TRUE |
| USE_OMP | Whether or not to compile with OpenMP multithreading. Sucks, don't use. | None. | None. | FALSE |
| USE_CUDA | Whether or not to compile with support for running on CUDA/graphics cards. There isn't much optimization for this yet, probably not worth using. | None. | None. | FALSE |
| NO_MPI_CHECKING | No idea. | None. | None. | FALSE |

## Compilation on the HLRN

Currently (as of Feb 8, 2024) we load the following modules
```
export FFTW_DIR="/sw/numerics/fftw3/impi/intel/3.3.8/skl/lib"
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH

module load intel impi/2019.5 fftw3/impi/intel/3.3.8 hdf5-parallel/impi/intel/1.10.5 gcc/9.3.0
```
The GNUMakefile and make files in amrex are already configured for use on the HLRN (AMREX_HOME might need to be adjusted depending on the location of the amrex directory). It is important to include the fftw3 library via 
```
LIBRARIES += -L$(FFTW_DIR) -lfftw3_mpi -lfftw3
``` 
Moreover, a file called ```Make.local``` needs to be created in ```/amrex/Tools/GNUMake``` with the following content:
```
CXX := mpicxx
CC := mpicc
FC := mpif90
F90 := mpif90

```
## Compilation on the gwdg

Currently (as of Feb 8, 2024) we load the following modules 

```
export FFTW_DIR='/opt/sw/rev/23.12/linux-scientific7-cascadelake/gcc-9.5.0/fftw-3.3.8-62yh7t/lib'
export LD_LIBRARY_PATH=$FFTW_DIR:$LD_LIBRARY_PATH

module load openmpi/4.1.6 gcc/9.5.0 fftw/3.3.10 hdf5/1.14.3 python/3.9.0

```
The GNUMakefile and make files in amrex are already configured for use on the HLRN (AMREX_HOME might need to be adjusted depending on the location of the amrex directory). It is important to include the fftw3 library via 
```
LIBRARIES += -L$(FFTW_DIR) -lfftw3_omp -lfftw3
``` 
Moreover, a file called ```Make.local``` needs to be created in ```/amrex/Tools/GNUMake``` with the following content:
```
CXX := mpicxx
CC := mpicc
FC := mpif90
F90 := mpif90

```


## Input Parameters

### Axionyx_KG Input Parameters

| Flag | Description | Conflicts | Default |
| --- | --- | --- | --- |
| ax.minkowski | Turns on/off evolution of the scale factor. Int; 0 for false, anything else for true. | None. | 0 |
| KG.simType | Model to use. Single Field Quartic = 0, Generalised Monodromy = 1, Pseudo-Lorentzian = 2 | None. | Must be set |
| KG.ICType | Initial conditions to use.  Uniform value = 0, fixed k sine wave = 1, delta k in momentum-space = 2, standard post-inflation = 3 | None | Must be set | 
| KG.KG_k | Fixed momentum mode for either fixed k sine wave or delta k. | None. | 1 |
| KG.cutoff_k | For standard inflationary ICs, cut off all momentum modes with $i^2 + j^2 + k^2 > \text{cutoff\_k}^2$. A value of 0 disables the cutoff. | None. | 0 |
| KG.KG0| Initial field value in all models | None. | 0 |
| KG.mass | $m$ in all models | None. | Must be set |
| KG.lambda | $\lambda$ (quartic coupling) in quartic model. | None. | Must be set |
| KG.MASS | $M$ in monodromy and pseudo-Lorentzian models | None. | Must be set |
| KG.power | $\alpha$ in monodromy model | None. | Must be set |


### Nyx Input Parameters

| Flag | Description | Conflicts | Default |
| --- | --- | --- | --- |
| nyx.comoving_OmB | $\Omega_B/\Omega_{\text{crit}}$, the fraction of baryon energy density to the critical energy density. Allowed values: [0.,1.]. Ignored when USE_INFLATION=TRUE.  | None. | 0. |
| nyx.comoving_OmM | $\Omega_M/\Omega_{\text{crit}}$, the fraction of total matter (baryon+dark matter) energy density to the critical energy density. Allowed values: [0.,1.]. Ignored when USE_INFLATION=TRUE.  | None. | 0. |
| nyx.comoving_OmR | $\Omega_R/\Omega_{\text{crit}}$, the fraction of radiation energy density to the critical energy density. Allowed values: [0.,1.]. Ignored when USE_INFLATION=TRUE.  | None. | 0. |

### AMReX Input Parameters

| Flag | Description | Conflicts | Default |
| --- | --- | --- | --- |
| geometry.is_periodic | Sets whether or not the lattice has periodic boundary conditions. Int; 0 for false, 1 for true. Accepts one value to set all dimensions the same, or one value for each dimension, e.g., "1" or "1 1 0". | None. | 0 |
| geometry.coord_sys | Sets the coordinate system to use. Int; 0 for Cartesian, 1 for polar [CHECK THIS]. | None. | 0 |
| geometry.prob_lo | Sets the lower boundary. Float. Accepts one value to set all dimensions the same, or one value for each dimension. | None. | 0. |
| geometry.prob_hi | Sets the upper boundary. Float. Accepts one value to set all dimensions the same, or one value for each dimension. | None. | 0. |
| amr.n_cell | Sets the initial course grid number of cells. Int. Accepts one value to set all dimensions the same, or one value for each dimension. | None. | 64 |



---

***Original Nyx README below.***

---


# Nyx

[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)

*An adaptive mesh, massively-parallel, cosmological simulation code*

******

### About

Nyx code solves equations of compressible hydrodynamics on an adaptive grid
hierarchy coupled with an N-body treatment of dark matter. The gas dynamics in
Nyx uses a finite volume methodology on an adaptive set of 3-D Eulerian grids;
dark matter is represented as discrete particles moving under the influence of
gravity. Particles are evolved via a particle-mesh method, using Cloud-in-Cell
deposition/interpolation scheme. Both baryonic and dark matter contribute to
the gravitational field. In addition, Nyx currently includes physics needed to
accurately model the intergalactic medium: in optically thin limit and assuming
ionization equilibrium, the code calculates heating and cooling processes of the
primordial-composition gas in an ionizing ultraviolet background radiation field.
Additional physics capabilities are under development.

While Nyx can run on any Linux system in general, we particularly focus on supercomputer systems.
Nyx is parallelized with MPI + X, where "X" can be OpenMP, CUDA, or HIP (DPC++ implementation
is ongoing). In the OpenMP regime, Nyx and has been successfully run at parallel concurrency
of up to 2,097,152 (on NERSC's Cori-KNL). With Cuda implementation, it was ran on up to
13,824 GPUs (on OLCF's Summit).

More information on Nyx can be found at 
http://amrex-astro.github.io/Nyx/

### Standards and dependencies

To compile the code we require C++11 compliant compilers that support MPI-2 or
higher implementation.  If threads or accelerators are used, we require 
OpenMP 4.5 or higher, Cuda 9 or higher, or HIP-Clang.

To use Nyx, you also need AMReX:
https://github.com/AMReX-codes/amrex

For example, to compile the Lyman alpha (LyA) executable on Summit:
```sh
$ module load gcc/6.4.0 cuda/11.0.3

$ git clone https://github.com/AMReX-Codes/amrex.git
$ git clone https://github.com/AMReX-astro/Nyx.git

$ cd Nyx/Exec/LyA
$ make -j 12 USE_CUDA=TRUE
```


There is a User's Guide in `Nyx/UsersGuide/` (type `make` to build
from LaTeX source) that will guide you through running your first
problem.


### Development model

The `development` branch in also the main branch.  We use nightly
regression testing to ensure that no answers change (or if they do, that
the changes were expected). Contributions are welcomed and should be done via pull requests.
A pull request should be generated from your fork of Nyx and should target
the `development` branch.


### Physics

For the description of the N-body and adiabatic hydro algorithms in Nyx, see
Almgren, Bell, Lijewski, Lukic & Van Andel (2013), ApJ, 765, 39:
http://adsabs.harvard.edu/abs/2013ApJ...765...39A

For the reaction and thermal rates of the primordial chemical composition gas 
(and convergence tests in the context of the Lyman-alpha forest), see
Lukic, Stark, Nugent, White, Meiksin & Almgren (2015), MNRAS, 446, 3697:
http://adsabs.harvard.edu/abs/2015MNRAS.446.3697L

For considerations regarding the spatially uniform synthesis model of the UV background, 
which provides the photo-ionization and photo-heating rates, see Onorbe,
Hennawi & Lukic (2017), ApJ, 837, 106:
http://adsabs.harvard.edu/abs/2017ApJ...837..106O

We have also implemented non-radiative transfer methods to model inhomogeneous reionization,
the paper is in preparation.


### Output

Nyx outputs certain global diagnostics at each timestep and plot files at regular
intervals, or at user-specified redshifts. Visualization packages
[VisIt](https://wci.llnl.gov/simulation/computer-codes/visit),
[Paraview](https://www.paraview.org/)
and [yt](http://yt-project.org/)
have built-in support for the AMReX file format used by Nyx.

In addition, Nyx interfaces with two post-processing suites, Reeber and Gimlet. Reeber
uses topological methods to construct merge trees of scalar fields, which Nyx in
turn uses to find halos. Gimlet computes a variety of quantities
related to the Lyman-alpha forest science. These suites are fully MPI-parallel and can
be run either "in situ" or "in-transit", or with a combination of both.


### License
Nyx is released under the LBL's modified BSD license, see the [license.txt](license.txt) file for details.


### Contact

For questions, comments, suggestions, contact Ann Almgren at ASAlmgren@lbl.gov
or Zarija Lukic at zarija@lbl.gov .
