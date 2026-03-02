#ifndef _COMOV_FULL_H_
#define _COMOV_FULL_H_

// There can be only one.
#ifdef _COMOV_EOS_H_
#error Cannot compile with both COMOV_FULL and COMOV_EOS.
#endif // _COMOV_EOS_H_

#include <AMReX_REAL.H>

////////////////////////////////////////////
//                                        //
//  A set of utilities for tracking       //
//  comoving coordinates based on a fixed //
//  equation of state.                    //
//                                        //
//  For a self-consistent evolution of    //
//  the scale factor, use Comoving_Full.  //
//                                        //
////////////////////////////////////////////

namespace Comoving {
extern amrex::Real final_a;
//
// Read inputs related to comoving coordinates
//
void read_comoving_params();
void initComov(amrex::Real A, amrex::Real B, amrex::Real s, amrex::Real r,
               amrex::Real V0, int gridvol);
// If we're using MPI, we also need to free the shared MPI windows for the
// energy densities
void restartComov(amrex::Real A, amrex::Real B, amrex::Real s, amrex::Real r,
                  int gridvol, amrex::Real a, amrex::Real ap, amrex::Real app,
                  amrex::Real a_prev, amrex::Real ap_prev, amrex::Real app_prev,
                  amrex::Real time, amrex::Real prev_time);

//
// Get the comoving coordinate "a" and its derivatives (all in program units)
//
amrex::Real get_comoving_a(amrex::Real tt = 0.);
amrex::Real get_comoving_ap(amrex::Real tt = 0.);
amrex::Real get_comoving_app(amrex::Real tt = 0.);

//
// Get the (un-normalized) energy densities (all in program units)
//
amrex::Real get_rho_g();
amrex::Real get_rho_v();
amrex::Real get_rho_t();

//
// Get the energy conservation debug ratio (~ H^2/rho)
//
amrex::Real debug_ratio();

//
// Set the initial ap based off of debug ratio, and calculate app
//
void set_ics();

// Evolve the scale factor
void kick_a(amrex::Real dt_half, bool first);
void drift_a(amrex::Real dt_half);

// Set and reset the stored values for the energy density
void reset_rho();
void add_to_rho(amrex::Real grad, amrex::Real pot, amrex::Real kin);
amrex::Real get_gridsize();

// Stoping the simulation if final_a is defined and reached
void stop_at_final_a(amrex::Real a_now, amrex::Real a_final);
} // namespace Comoving

#endif // _COMOV_FULL_H_
