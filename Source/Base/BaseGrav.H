#ifndef _Base_Grav_H_
#define _Base_Grav_H_

#include <AxSCHComov.H>
#include <BaseAx.H>
#include <Gravity.H>

// AMReX
// #include <AMReX.H> // Include AMReX headers as needed
// #include <AMReX_AmrLevel.H>
// #include <AMReX_Amr.H>
// #include <AMReX_Geometry.H>
// #include <AMReX_BoxArray.H>
// #include <AMReX_DistributionMapping.H>
// #include <AMReX_Vector.H> // Assuming AMReX vector is used
// #include <AMReX_MultiFab.H> // Assuming AMReX MultiFab is used

// #include <vector>  // Assuming std::vector is used

using namespace amrex;

// We don't use forward declarations for these. I'm iffy on the amrex above but
// I'll leave it for now. -PH
// // Forward declarations or necessary includes for particle types
// using std::istream;
// using std::ostream;

// Maximum problem parameters when hydrodynamics is included
// static constexpr int max_prob_param = 20;

class BaseGrav : public AxSCHComov {

public:
  BaseGrav();

  BaseGrav(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
           const amrex::BoxArray &ba, const amrex::DistributionMapping &dm,
           amrex::Real time, int dens = 0, int stype = 0, int ptype = 1,
           int gtype = 2);

  ~BaseGrav();

  static int reuse_mlpoisson;

  amrex::Real get_comoving_a(amrex::Real time) const;

  amrex::Real vol_weight_sum(const std::string &name, amrex::Real time,
                             bool masked = false);
  amrex::Real vol_weight_sum(const MultiFab &mf, bool masked = false);

  // Virtual function to retrieving a density field
  virtual MultiFab &get_density(bool old = false) = 0;

  // static void variable_setup();

  // protected:

  // void build_metrics();

  static Gravity *gravity;
};

#endif // BASE_GRAV
