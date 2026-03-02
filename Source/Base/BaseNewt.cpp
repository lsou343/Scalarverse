#include <AxKGComov.H> // LSR -- change to KG
#include <iostream>
#include <string>

#include <BaseAx.H>
#include <BaseNewt.H>
#include <Comoving_Full.H> // LSR -- changed to Comoving_Full.H
#include <Newtonian.H>

#include <constants_cosmo.H>

#include <AMReX_CONSTANTS.H>
#include <AMReX_Print.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#ifdef BL_USE_MPI
#include <MemInfo.H>
#endif

using namespace amrex;

// Initialize the static Gravity pointer
Gravity *BaseNewt::gravity = nullptr;

int BaseNewt::reuse_mlpoisson = 0;

BaseNewt::BaseNewt() { BL_PROFILE("BaseNewt::BaseNewt()"); }

BaseNewt::BaseNewt(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
                   const amrex::BoxArray &ba,
                   const amrex::DistributionMapping &dm, amrex::Real time,
                   int dens, int stype, int ptype, int gtype)
    : AxKGComov(papa, lev, level_geom, ba, dm, time) {
  BL_PROFILE("BaseNewt::BaseNewt(Amr)");

  // Gravity setup
  if (lev == 0) {
    if (!gravity) {
      // std::cout << "Initializing Gravity object." << std::endl;
      gravity = new Gravity(parent, parent->finestLevel(), &phys_bc, dens,
                            stype, ptype, gtype);
      if (!gravity) {
        amrex::Abort("Fatal Error: Gravity object allocation failed!");
      }
    }
  }

  // Always update the Gravity object with the current level data.
  // std::cout << "Installing gravity level " << lev << "..." << std::endl;
  gravity->install_level(lev, this);

  if (!gravity) {
    amrex::Abort("Fatal Error: Gravity object is NULL after initialization!");
  }

  // build_metrics();
}

// Destructor definition
BaseNewt::~BaseNewt() {}

// void BaseNewt::build_metrics() {}

amrex::Real BaseNewt::get_comoving_a(amrex::Real time) const {
  return Comoving::get_comoving_a(time);
}

// Computes the volume-weighted sum of a grid quantity specified by its name at
// a given time.
Real BaseNewt::vol_weight_sum(const std::string &name, amrex::Real time,
                              bool masked) {
  BL_PROFILE("BaseNewt::vol_weight_sum(name)");

  // std::cout
  //     << "BaseNewt::vol_weight_sum: Starting calculation for derived quantity
  //     '"
  //     << name << "' at time " << time << "." << std::endl;

  // Attempt to derive the quantity
  // std::cout << "Attempting to derive quantity '" << name << "' at time " <<
  // time
  // << std::endl;
  auto mf = derive(name, time, 0);
  if (!mf) {
    amrex::Abort(
        "vol_weight_sum: Derivation of '" + name +
        "' failed. Possible missing state descriptor or uninitialized field.");
  }

  Real sum = vol_weight_sum(*mf, masked);
  return sum;
}

// Calculates the volume-weighted sum directly on a given MultiFab, optionally
// applying a mask.
Real BaseNewt::vol_weight_sum(const MultiFab &mf, bool masked) {
  BL_PROFILE("BaseNewt::vol_weight_sum");

  // std::cout << "vol_weight_sum: Checking MultiFab validity." << std::endl;

  const auto dx = geom.CellSizeArray();

  MultiFab *mask = 0;
  if (masked) {
    int flev = parent->finestLevel();
    while (parent->getAmrLevels()[flev] == nullptr)
      flev--;

    if (level < flev) {
      BaseAx *fine_level =
          dynamic_cast<BaseAx *>(&(parent->getLevel(level + 1)));
      mask = fine_level->build_fine_mask();
    }
  }

  ReduceOps<ReduceOpSum> reduce_op;
  ReduceData<Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;

  if (!masked || (mask == 0)) {
    BL_PROFILE("Nyx::vol_weight_sum()::ReduceOpsOnDevice");
#ifndef AMREX_USE_GPU
#ifdef _OPENMP
#pragma omp parallel if (!system::regtest_reduction)
#endif
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const auto fab = mf.array(mfi);
      const Box &tbx = mfi.tilebox();

      reduce_op.eval(
          tbx, reduce_data,
          [fab] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
            Real x = fab(i, j, k);
            return x;
          });
    }

  } else {
    BL_PROFILE("Nyx::vol_weight_sum()::ReduceOpsOnDevice");
#ifndef AMREX_USE_GPU
#ifdef _OPENMP
#pragma omp parallel if (!system::regtest_reduction)
#endif
#endif
    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const auto fab = mf.array(mfi);
      const auto msk = mask->array(mfi);
      const Box &tbx = mfi.tilebox();

      reduce_op.eval(
          tbx, reduce_data,
          [fab, msk] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
            Real x = fab(i, j, k) * msk(i, j, k);
            return x;
          });
    }
  }

  ReduceTuple hv = reduce_data.value();
  ParallelDescriptor::ReduceRealSum(amrex::get<0>(hv));

  Real sum = get<0>(hv) * (dx[0] * dx[1] * dx[2]);

  if (!masked)
    sum /= geom.ProbSize();

  // std::cout << "vol_weight_sum: Computation complete. Volume-weighted sum = "
  // << sum << std::endl;
  return sum;
}

// void BaseNewt::variable_setup()
// {
//     BaseAx::variable_setup();

//     // Get gravity-specific options
//     gravity->read_params();

// }
