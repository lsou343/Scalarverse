#include <AMReX_LevelBld.H>

// Select your simulation here. Just add your compile flag and change PREF
// appropriately.
#ifdef INFLATION
#ifdef COMOV_FULL
#include <AxKGComov.H>
#define PREF AxKGComov
#else
#include <AxKG.H>
#define PREF AxKG
#endif
#endif

#ifdef SCHROEDINGER
#ifdef COMOV_EOS
#ifdef GRAV
#include <AxSP.H>
#define PREF AxSP
#else
#include <AxSCHComov.H>
#define PREF AxSCHComov
#endif
#else
#include <AxSCH.H>
#define PREF AxSCH
#endif
#endif

#ifdef GRAV
#ifndef SCHROEDINGER
#include <AxTestGrav.H>
#define PREF AxTestGrav
#endif // not SCHROEDINGER
#endif // GRAV
using namespace amrex;

class AxBld : public LevelBld {
  virtual void variable_setup();
  virtual void variable_cleanup();

  // hack copies for amrex overriding. ... But... why?
  virtual void variableSetUp();
  virtual void variableCleanUp();

  virtual AmrLevel *operator()();
  virtual AmrLevel *operator()(Amr &papa, int lev, const Geometry &level_geom,
                               const BoxArray &ba,
                               const DistributionMapping &dm, Real time);
};

AxBld Ax_bld;

LevelBld *get_level_bld() { return &Ax_bld; }

void AxBld::variable_setup() { PREF::variable_setup(); }

void AxBld::variable_cleanup() { PREF::variable_cleanup(); }

AmrLevel *AxBld::operator()() { return new PREF; }

AmrLevel *AxBld::operator()(Amr &papa, int lev, const Geometry &level_geom,
                            const BoxArray &ba, const DistributionMapping &dm,
                            Real time) {
  return new PREF(papa, lev, level_geom, ba, dm, time);
}

// override hacks, copies of above
LevelBld *getLevelBld() { return &Ax_bld; }

void AxBld::variableSetUp() { PREF::variable_setup(); }

void AxBld::variableCleanUp() { PREF::variable_cleanup(); }
