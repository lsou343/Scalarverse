#ifndef COMOVING_EOS_H_
#define COMOVING_EOS_H_
#include <AMReX_REAL.H>
using namespace amrex;
namespace Comoving {
// Read parameters (from ParmParse) and compute initial and final times.
extern Real initial_time;
extern Real comoving_OmM;
extern Real comoving_h;
extern Real final_a;
extern Real initial_a;

void read_comoving_params();

// Initialize the comoving state at the beginning of the simulation.
// For level==0 and time==0, if the state has not been initialized,
// set old_a_time and new_a_time to 0, and old_a and new_a to 1/(1+initial_z).
void init_comoving_state(Real time, int level);

// timestepping
Real comoving_est_time_step(Real time, amrex::Real &dt);

// Advance the comoving state by a time step dt.
void comoving_update_a_integrate(Real time, amrex::Real dt, int level);

// Return the comoving scale factor for a given simulation time.
Real get_comoving_a(amrex::Real time);

} // namespace Comoving
#endif // COMOVING_EOS_H_
