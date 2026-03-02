#include <AMReX_MultiFabUtil.H>
#include <Comoving_EOS.H>

using namespace amrex;
#ifdef GRAV
#include <AxSP.H>
#include <Gravity.H>
#endif
#include <AxSCHComov.H>

Real AxSCHComov::advance(Real time, Real dt, int iteration, int ncycle) {
  // #endif
  // Creates a tag be able to track how long it takes
  MultiFab::RegionTag amrlevel_tag("AmrLevel_Level_" + std::to_string(level));

  // tracks performance of the function
  BL_PROFILE("AxSCHComov::advance_SCHComov()");
  Gpu::LaunchSafeGuard lsg(true);

// Move newData to oldData to be able to overwrite newData
#ifdef GRAV
  for (int k = 0; k < AxSP::nStates(); k++) {
#else
  for (int k = 0; k < AxSCH::nStates(); k++) {
#endif
    state[k].allocOldData();
    state[k].swapTimeLevels(dt);
  }
  if (verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "Advancing the Sch field at level " << level << " ...\n";
  }

  // // checking if gravity is defined by printing the maximum of phi
  // Real maxphi =
  //     get_new_data(AxSP::getState(AxSP::StateType::PhiGrav_Type)).max(0);
  // Print() << "AxSP::advance: maxphi = " << maxphi << "\n";

#ifdef GRAV

  const int finest_level = parent->finestLevel();
  BL_PROFILE_VAR("solve_for_old_phi", solve_for_old_phi);
  // if (level == 0 || iteration > 1) {
  MultiFab::RegionTag amrGrav_tag("Gravity_" + std::to_string(level));
  for (int lev = level; lev < finest_level; lev++) {
    BaseGrav::gravity->zero_phi_flux_reg(lev + 1);
    Print() << "AxSP::advance: zeroing phi flux reg at level " << lev + 1
            << "\n";
  }
  // swap grav data
  for (int lev = level; lev <= finest_level; lev++) {
    // get_level(lev).
    BaseGrav::gravity->swap_time_levels(lev);
  }
  // Solve for phi using the previous phi as a guess.
  int use_previous_phi_as_guess = 1;
  int ngrow_for_solve = 1; // iteration + stencil_deposition_width;
  BaseGrav::gravity->multilevel_solve_for_old_phi(
      level, finest_level, ngrow_for_solve, use_previous_phi_as_guess);
  // }
  BL_PROFILE_VAR_STOP(solve_for_old_phi);
#endif

  // why not using the time variable instead of prev_time and cur_time?
  const Real prev_time = state[getState(StateType::SCH_Type)].prevTime();
  const Real cur_time = state[getState(StateType::SCH_Type)].curTime();
  const Real a_old = Comoving::get_comoving_a(prev_time);
  const Real a_new = Comoving::get_comoving_a(cur_time);

  if (level == 0 && AxSCH::PSorFD == 0) {
    // amrex::Abort("Pseudo Spectral method is implemented but doesnt work
    // yet.
    // "
    //              "Please fix it or use a different SCH.PSorFD (look at the
    //              " "input file) on the first level!");

    AxSCHComov::advance_SCH_PS(time, dt, a_old, a_new);
    // Print() << "advance at level " << level << " using PS" << "\n";
  } else {
    AxSCHComov::advance_SCH_FD(time, dt, a_old, a_new);
    // Print() << "advance at level " << level << " using FD" << "\n";
  }

#ifdef GRAV
  MultiFab::Copy(parent->getLevel(level).get_new_data(
                     AxSP::getState(AxSP::StateType::PhiGrav_Type)),
                 parent->getLevel(level).get_old_data(
                     AxSP::getState(AxSP::StateType::PhiGrav_Type)),
                 0, 0, 1, 0);

  // Solve for new Gravity

  BL_PROFILE_VAR("solve_for_new_phi", solve_for_new_phi);
  // int use_previous_phi_as_guess = 1;

  // MultiFab::RegionTag amrGrav_tag("Gravity_" + std::to_string(level));
  int fill_interior = 0;
  int grav_n_grow = 1;
  BaseGrav::gravity->solve_for_new_phi(
      level, get_new_data(AxSP::getState(AxSP::StateType::PhiGrav_Type)),
      BaseGrav::gravity->get_grad_phi_curr(level), fill_interior, grav_n_grow);

  BL_PROFILE_VAR_STOP(solve_for_new_phi);
#endif

  return dt;
}
