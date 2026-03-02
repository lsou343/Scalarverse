#include <AxSCHComov.H>
#include <Comoving_EOS.H>

amrex::Real change_max = 1.1;

//
// Destructor
//
AxSCHComov::~AxSCHComov() {
  // Clean up any resources if needed
}

//
// Setup any variables/states needed for a comoving run
// (overrides AxSCH::variable_setup if you want to *add* more descriptors)
//
void AxSCHComov::variable_setup() {
  // First, reuse AxSCH’s variable setup
  AxSCH::variable_setup();

  // amrex::Real cur_time = state[State_for_Time].curTime();
  // Comoving::init_comoving_state(cur_time, level);

  Comoving::read_comoving_params();

  // Then do additional comoving-specific setup if needed:
  // e.g. define new derived variables or store a(t), etc.
  // ...
}

//
// Called when the level is initialized from a coarser, already existing level
//
void AxSCHComov::init(amrex::AmrLevel &old) {
  // First call the base SCH init
  AxSCH::init(old);

  // Get the current time from the state data
  // amrex::Real current_time = state[State_for_Time].curTime();
  // Comoving::init_comoving_state(current_time, level);

  // Possibly read comoving data from old if you want continuity
  // ...
}

//
// Called when a *new* level is made (e.g., after regridding)
//
void AxSCHComov::init() {
  // Base initialization for AxSCH
  AxSCH::init();

  // Possibly do comoving-specific logic for newly created level
  // ...
}

//
// Initialize data on this level (including the wavefunction, etc.)
//
void AxSCHComov::initData() {
  BL_PROFILE("AxSCHComov::initData()");

  amrex::Gpu::LaunchSafeGuard lsg(true);
  // Here we initialize the grid data and the particles from a plotfile.
  if (!parent->theRestartPlotFile().empty()) {
    amrex::Abort(
        "AmrData requires fortran"); // This is a bizarre error message...
    return;
  }

  if (verbose && amrex::ParallelDescriptor::IOProcessor())
    amrex::Print() << "Initializing the data at level " << level << '\n';

  const auto dx = geom.CellSizeArray();
  const auto geomdata = geom.data();

  // Make sure dx = dy = dz -- that's all we guarantee to support
  const amrex::Real SMALL = 1.e-13;
  if ((fabs(dx[0] - dx[1]) > SMALL) || (fabs(dx[0] - dx[2]) > SMALL))
    amrex::Abort("We don't support dx != dy != dz");

  amrex::MultiFab &SCH_new = get_new_data(getState(StateType::SCH_Type));
  amrex::GpuArray<amrex::Real, BaseAx::max_prob_param>
      prob_param; // Array of parameters required for initial conditions.
  prob_param.fill(PAR_ERR_VAL); // Fill it with error values first.
  prob_param_fill(prob_param);  // Add what values are needed.

  // We have to do things slightly differently if we're initializing in
  // position-space or Fourier space.
  if (ic == ICType::test || ic == ICType::plain_wave ||
      ic == ICType::gaussian || ic == ICType::linearperurbation) {
    for (amrex::MFIter mfi(SCH_new, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box &bx = mfi.tilebox();
      const auto fab_SCH_new = SCH_new.array(mfi);
      prob_initdata_pos_on_box(bx, fab_SCH_new, geomdata, prob_param);
    }
    SCH_new.FillBoundary(geom.periodicity());
  } else if (ic == ICType::KGfield) {
    // Initialize with Output of KG simulation
    int nghost = 0;
    MultiFab KG_init_mf;
    AxSCH::read_KG_MultiFab(KGinitDirName, nghost, KG_init_mf);
    amrex::Print() << "A = " << KGA << std::endl;
    amrex::Print() << "m = " << KGm << std::endl;

    Real a_pr, ap;
    // Get the correct scale factor and Hubble constant
    if (amrex::ParallelDescriptor::IOProcessor()) {

      // Format for Comoving_Full output file:
      // a
      // ap
      // app
      std::string FileName = KGinitDirName + "/comoving_a";
      std::ifstream File;
      File.open(FileName.c_str(), std::ios::in);
      if (!File.good())
        amrex::FileOpenFailed(FileName);
      File >> a_pr;
      File >> ap;

      amrex::Print() << "KG a = " << a_pr << std::endl;
      amrex::Print() << "H = " << KGB * ap * std::pow(a_pr, KGs - 1.)
                     << std::endl;
    }
    // Spread the good word
    ParallelDescriptor::Bcast(&a_pr, 1,
                              ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::Bcast(&ap, 1, ParallelDescriptor::IOProcessorNumber());

    // KG-field (arr_old(i,j,k,0)) and derivative (arr_old(i,j,k,1)) in KG
    // program units.

    const Real CR = KGm / (std::sqrt(2.) * KGA * std::pow(a_pr, KGr));
    const Real CI = KGB / KGA * std::pow(a_pr, KGs - KGr) / std::sqrt(2.);
    const Real coeff = 6058407795815140.0; // conversion factor from m_p^2 ->
                                           // sqrt(m_u/l_u^3) for wave function.
    Real comoving_a = Comoving::get_comoving_a(state[State_for_Time].curTime());

    // for (MFIter mfi(Ax_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    for (MFIter mfi(SCH_new, false); mfi.isValid(); ++mfi) {
      // const Box& bx = mfi.tilebox();
      const Box &bx = mfi.validbox();
      Array4<Real> const &arr_SCH_init = SCH_new.array(mfi);
      Array4<Real> const &arr_KG_init = KG_init_mf.array(mfi);
      ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        // AMREX_PARALLEL_FOR_3D(bx, i, j, k, {
        // Real field:
        arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) =
            coeff * CR * std::pow(comoving_a, 1.5) *
            arr_KG_init(
                i, j, k,
                0); // additional factor of a^1.5 to convert to comoving WF
        // Imaginary field:
        arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) =
            coeff * std::pow(comoving_a, 1.5) * CI *
            (arr_KG_init(i, j, k, 1) -
             KGr * ap / a_pr * arr_KG_init(i, j, k, 0));
        // Density:
        arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::Dens)) =
            arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) *
                arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) +
            arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) *
                arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im));
        // Phase:
        arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::Phase)) =
            std::atan2(
                arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)),
                arr_SCH_init(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)));
      });
    }
    // #TODO
    SCH_new.FillBoundary(geom.periodicity());

  } else {
#ifdef BL_USE_MPI
    prob_initdata_mom_on_mf(SCH_new, geom, prob_param);
    SCH_new.plus(
        1., 0, 1); // (val, comp, ncomp): Adds the value val to ncomp components
                   // starting at comp. Note f_pr starts at 1 and f_pr = f/SCH0.
    SCH_new.FillBoundary(geom.periodicity());
#endif // BL_USE_MPI
  }
}

//
// Compute the next time step, possibly factoring in comoving constraints
//
amrex::Real AxSCHComov::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxSCHComov::est_time_step()");
  // In case we have a simple fixed time step.
  if (BaseAx::fixed_dt > 0)
    return BaseAx::fixed_dt;

  amrex::Real est_dt = 1.0e+200;

  if (vonNeumann_dt > 0) {

    amrex::Real cur_time = state[State_for_Time].curTime();
    amrex::Real a = Comoving::get_comoving_a(cur_time);
    // amrex::Real a = 1;
    const amrex::Real *dx = geom.CellSize();
    amrex::Real dt_cfl = dx[0] * dx[0] * a * a / 6.0 /
                         hbaroverm; // stability condition for the SCH equation
#ifdef GRAVITY
    const amrex::MultiFab &phi = get_new_data(PhiGrav_Type);
    amrex::Real phi_max = std::abs(phi.max(0) - phi.min(0));
    dt_cfl = std::min(dt_cfl, hbaroverm / phi_max);
#endif
    if (vonNeumann_dt < 1.0)
      dt_cfl *= vonNeumann_dt;

    // if (levelmethod[level]==PSlevel)
    //   dt_cfl *= vonNeumann_dt;
    // else if (vonNeumann_dt<1.0)
    if (dt_old > 0)
      est_dt = std::min(dt_old, dt_cfl);
    else
      est_dt = dt_cfl;

    if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print() << "AxSCHComov::est_time_step at level " << level
                     << ":  est_dt = " << est_dt << "\n";
    }
  }
  // amrex::Print() << "AxSCHComov::cfl est_time_step at level " << level
  //                << ":  est_dt = " << est_dt << "\n";

  // Comoving est_time_step
  if (level == 0) {
    amrex::Real cur_time = state[State_for_Time].curTime();
    est_dt = Comoving::comoving_est_time_step(cur_time, est_dt);
  }
  // amrex::Print() << "AxSCHComov::comoving est_time_step at level " << level
  //                << ":  est_dt = " << est_dt << "\n";

  return est_dt;
}

// computeNewDt is different to BaseAx, due to inclusion of comoving_a
void AxSCHComov::computeNewDt(int finest_level, int sub_cycle,
                              amrex::Vector<int> &n_cycle,
                              const amrex::Vector<amrex::IntVect> &ref_ratio,
                              amrex::Vector<amrex::Real> &dt_min,
                              amrex::Vector<amrex::Real> &dt_level,
                              amrex::Real stop_time, int post_regrid_flag) {
  BL_PROFILE("Comoving_EOS::computeNewDt()");
  //
  // We are at the start of a coarse grid timecycle.
  // Compute the timesteps for the next iteration.
  //
  if (level > 0)
    return;

  int i;

  amrex::Real dt_0 = 1.0e+100;
  int n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    AxSCHComov &adv_level = static_cast<AxSCHComov &>(get_level(i));
    dt_min[i] = adv_level.est_time_step(dt_level[i]);
  }

  if (fixed_dt <= 0.0) {
    if (post_regrid_flag == 1) {
      //
      // Limit dt's by pre-regrid dt
      //
      for (i = 0; i <= finest_level; i++) {
        dt_min[i] = std::min(dt_min[i], dt_level[i]);
      }
      //
      // Find the minimum over all levels
      //
      for (i = 0; i <= finest_level; i++) {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_min[i]);
      }
    } else {
      bool sub_unchanged = true;
      if ((parent->maxLevel() > 0) && (level == 0) &&
          (parent->subcyclingMode() == "Optimal") &&
          (parent->okToRegrid(level) || parent->levelSteps(0) == 0)) {
        int new_cycle[finest_level + 1];
        for (i = 0; i <= finest_level; i++)
          new_cycle[i] = n_cycle[i];
        // The max allowable dt
        amrex::Real dt_max[finest_level + 1];
        for (i = 0; i <= finest_level; i++) {
          dt_max[i] = dt_min[i];
        }
        // find the maximum number of cycles allowed.
        int cycle_max[finest_level + 1];
        cycle_max[0] = 1;
        for (i = 1; i <= finest_level; i++) {
          cycle_max[i] = parent->MaxRefRatio(i - 1);
        }
        // estimate the amout of work to advance each level.
        amrex::Real est_work[finest_level + 1];
        for (i = 0; i <= finest_level; i++) {
          est_work[i] = parent->getLevel(i).estimateWork();
        }
        // this value will be used only if the subcycling pattern is changed.
        dt_0 = parent->computeOptimalSubcycling(finest_level + 1, new_cycle,
                                                dt_max, est_work, cycle_max);
        for (i = 0; i <= finest_level; i++) {
          if (n_cycle[i] != new_cycle[i]) {
            sub_unchanged = false;
            n_cycle[i] = new_cycle[i];
          }
        }
      }

      if (sub_unchanged)
      //
      // Limit dt's by change_max * old dt
      //
      {
        for (i = 0; i <= finest_level; i++) {
          if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
            if (dt_min[i] > change_max * dt_level[i]) {
              std::cout << "BaseAx::compute_new_dt : limiting dt at level " << i
                        << '\n';
              std::cout << " ... new dt computed: " << dt_min[i] << '\n';
              std::cout << " ... but limiting to: " << change_max * dt_level[i]
                        << " = " << change_max << " * " << dt_level[i] << '\n';
            }
          }

          dt_min[i] = std::min(dt_min[i], change_max * dt_level[i]);
        }
        //
        // Find the minimum over all levels
        //
        for (i = 0; i <= finest_level; i++) {
          n_factor *= n_cycle[i];
          dt_0 = std::min(dt_0, n_factor * dt_min[i]);
        }
      } else {
        if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
          std::cout << "BaseAx: Changing subcycling pattern. New pattern:\n";
          for (i = 1; i <= finest_level; i++)
            std::cout << "   Lev / n_cycle: " << i << " " << n_cycle[i] << '\n';
        }
      }
    }
  } else {
    dt_0 = fixed_dt;
  }

  //
  // Limit dt's by the value of stop_time.
  //
  const amrex::Real eps = 0.001 * dt_0;
  amrex::Real cur_time = state[State_for_Time].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  // update scale factor
  Comoving::comoving_update_a_integrate(cur_time, dt_0, level);

  n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / n_factor;
  }
}
// computeInitialDt is different to BaseAx, due to inclusion of comoving_a
void AxSCHComov::computeInitialDt(
    int finest_level, int sub_cycle, amrex::Vector<int> &n_cycle,
    const amrex::Vector<amrex::IntVect> &ref_ratio,
    amrex::Vector<amrex::Real> &dt_level, amrex::Real stop_time) {
  BL_PROFILE("BaseAx::computeInitialDt()");
  // Grids have been constructed, compute dt for all levels.
  if (level > 0)
    return;

  int i;
  amrex::Real dt_0 = 1.0e+100;
  int n_factor = 1;

  if (parent->subcyclingMode() == "Optimal") {
    int new_cycle[finest_level + 1];
    for (i = 0; i <= finest_level; i++)
      new_cycle[i] = n_cycle[i];

    amrex::Real dt_max[finest_level + 1];
    for (i = 0; i <= finest_level; i++) {
      // Cast to BaseComovEOS so that the protected member is accessible.
      AxSCHComov &level_i = static_cast<AxSCHComov &>(get_level(i));
      dt_max[i] = level_i.initial_time_step();
    }
    // Find the maximum number of cycles allowed.
    int cycle_max[finest_level + 1];
    cycle_max[0] = 1;
    for (i = 1; i <= finest_level; i++) {
      cycle_max[i] = parent->MaxRefRatio(i - 1);
    }
    // Estimate the amount of work to advance each level.
    amrex::Real est_work[finest_level + 1];
    for (i = 0; i <= finest_level; i++) {
      est_work[i] = parent->getLevel(i).estimateWork();
    }
    dt_0 = parent->computeOptimalSubcycling(finest_level + 1, new_cycle, dt_max,
                                            est_work, cycle_max);
    for (i = 0; i <= finest_level; i++) {
      n_cycle[i] = new_cycle[i];
    }
    if (verbose && amrex::ParallelDescriptor::IOProcessor() &&
        finest_level > 0) {
      std::cout << "BaseAx: Initial subcycling pattern:\n";
      for (i = 0; i <= finest_level; i++)
        std::cout << "Level " << i << ": " << n_cycle[i] << '\n';
    }
  } else {
    for (i = 0; i <= finest_level; i++) {
      // Again, cast get_level(i) to BaseComovEOS to access initial_time_step.
      AxSCHComov &level_i = static_cast<AxSCHComov &>(get_level(i));
      dt_level[i] = level_i.initial_time_step();
      n_factor *= n_cycle[i];
      dt_0 = std::min(dt_0, n_factor * dt_level[i]);
    }
  }
  //
  // Limit dt's by the value of stop_time.
  //
  const amrex::Real eps = 0.001 * dt_0;
  amrex::Real cur_time = state[State_for_Time].curTime();
  if (stop_time >= 0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / n_factor;
  }
  Comoving::comoving_update_a_integrate(cur_time, dt_0, level);
}
//
// Averages down finer levels to coarser levels, if needed
//
void AxSCHComov::average_down() {
  // Reuse AxSCH logic, or add extra comoving fields if you have them
  AxSCH::average_down();
}

//
// Called after restarting a run from disk, so you can recover
// any comoving background variables (scale factor, etc.)
//
// void AxSCHComov::post_restart()
// {
//     // If not the finest or not level 0, do nothing special
//     if (level > 0) return;
//
//     // 1) Read in a(t), a'(t), etc. from your checkpoint file
//     //    e.g. from a file "comoving_a"
//     // 2) Store those in your comoving data structure
//     // 3) Optionally recalculate wavefunction norms, fill density, etc.
//     // ...
//     // fill_rho();
// }

//
// (Optional) Example of computing wavefunction density or other derived
// comoving quantity across the domain
//

//
// // (Optional) If you want to write out additional info to the job info or
// // separate logs for comoving data
// //
// void AxSCHComov::write_info() {
//   AxSCH::write_info();
//
//   // Then add comoving-specific prints/logging
//   // e.g. store a(t), wavefunction mass, etc.
// }
//
// //
// // (Optional) Called after writing a plotfile to disk
// //
// void AxSCHComov::writePlotFilePost(const std::string &dir, std::ostream &os)
// {
//   AxSCH::writePlotFilePost(dir, os);
//
//   // Save any comoving-specific data or diagnostics
//   // e.g. a(t) in a small ASCII file
// }

//
// (Optional) If you want to store comoving data in checkpoint
//
// void AxSCHComov::checkPointPost(const std::string &dir, std::ostream &os)
// {
//     // If AxSCH implements some checkPointPost, reuse it
//     // AxSCH::checkPointPost(dir, os);
//
//     // Then write out a(t), a'(t), etc. in "dir + /comoving_a"
//     // ...
// }
//
