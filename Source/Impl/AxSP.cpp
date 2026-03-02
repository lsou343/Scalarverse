#include <AxSP.H>
#include <bc_fill.H>
using namespace amrex;
namespace {
amrex::Real change_max = 1.1;

int scalar_bc[] = {
    // INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN, REFLECT_EVEN
    amrex::BCType::int_dir,      amrex::BCType::ext_dir,
    amrex::BCType::foextrap,     amrex::BCType::reflect_even,
    amrex::BCType::reflect_even, amrex::BCType::reflect_even};

int norm_vel_bc[] = {
    // INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_ODD, REFLECT_ODD, REFLECT_ODD
    amrex::BCType::int_dir,     amrex::BCType::ext_dir,
    amrex::BCType::foextrap,    amrex::BCType::reflect_odd,
    amrex::BCType::reflect_odd, amrex::BCType::reflect_odd};

int tang_vel_bc[] = {
    // INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN, REFLECT_EVEN
    amrex::BCType::int_dir,      amrex::BCType::ext_dir,
    amrex::BCType::foextrap,     amrex::BCType::reflect_even,
    amrex::BCType::reflect_even, amrex::BCType::reflect_even};

void set_scalar_bc(amrex::BCRec &bc, const amrex::BCRec &phys_bc) {
  const int *lo_bc = phys_bc.lo();
  const int *hi_bc = phys_bc.hi();
  for (int i = 0; i < BL_SPACEDIM; i++) {
    bc.setLo(i, scalar_bc[lo_bc[i]]);
    bc.setHi(i, scalar_bc[hi_bc[i]]);
  }
}

void set_x_vel_bc(BCRec &bc, const BCRec &phys_bc) {
  const int *lo_bc = phys_bc.lo();
  const int *hi_bc = phys_bc.hi();
  bc.setLo(0, norm_vel_bc[lo_bc[0]]);
  bc.setHi(0, norm_vel_bc[hi_bc[0]]);
  bc.setLo(1, tang_vel_bc[lo_bc[1]]);
  bc.setHi(1, tang_vel_bc[hi_bc[1]]);
  bc.setLo(2, tang_vel_bc[lo_bc[2]]);
  bc.setHi(2, tang_vel_bc[hi_bc[2]]);
}
void set_y_vel_bc(BCRec &bc, const BCRec &phys_bc) {
  const int *lo_bc = phys_bc.lo();
  const int *hi_bc = phys_bc.hi();
  bc.setLo(0, tang_vel_bc[lo_bc[0]]);
  bc.setHi(0, tang_vel_bc[hi_bc[0]]);
  bc.setLo(1, norm_vel_bc[lo_bc[1]]);
  bc.setHi(1, norm_vel_bc[hi_bc[1]]);
  bc.setLo(2, tang_vel_bc[lo_bc[2]]);
  bc.setHi(2, tang_vel_bc[hi_bc[2]]);
}

void set_z_vel_bc(BCRec &bc, const BCRec &phys_bc) {
  const int *lo_bc = phys_bc.lo();
  const int *hi_bc = phys_bc.hi();
  bc.setLo(0, tang_vel_bc[lo_bc[0]]);
  bc.setHi(0, tang_vel_bc[hi_bc[0]]);
  bc.setLo(1, tang_vel_bc[lo_bc[1]]);
  bc.setHi(1, tang_vel_bc[hi_bc[1]]);
  bc.setLo(2, norm_vel_bc[lo_bc[2]]);
  bc.setHi(2, norm_vel_bc[hi_bc[2]]);
}
} // namespace

// int BaseAx::NUM_STATE = AxSCH::nStates() + AxSP::nStates() override;

AxSP::AxSP() {
  BL_PROFILE("AxSP::AxTestGrav()");
  fine_mask = 0;
  // std::cout << "AxSP default constructor called." << std::endl;
}

AxSP::AxSP(Amr &papa, int lev, const Geometry &level_geom, const BoxArray &ba,
           const DistributionMapping &dm, Real time)
    : BaseGrav(papa, lev, level_geom, ba, dm, time,
               AxSCH::getField(AxSCH::Fields::Dens),
               AxSCH::getState(AxSCH::StateType::SCH_Type),
               getState(StateType::PhiGrav_Type),
               getState(StateType::Gravity_Type)) {
  BL_PROFILE("AxSP::AxTestGrav()");

  // std::cout << "AxSP constructor called for level " << lev << " at time "
  // << time << "." << std::endl;

  // if (level == 0 && time == 0.0) {
  // std::cout << "Initializing time-dependent variables." << std::endl;
  // }
} //
// Destructor
//
AxSP::~AxSP() {
  // Clean up any resources if needed
  //  if (fine_mask) {
  delete fine_mask;
  fine_mask = nullptr;
}
//
// Setup any variables/states needed for a comoving run
// (overrides AxSCHComov::variable_setup if you want to *add* more descriptors)
//
void AxSP::variable_setup() {
  // First, reuse AxSCHComov’s variable setup
  AxSCHComov::variable_setup();
  // Comoving::read_comoving_params();
  // Gravity::read_params();

  // Note that the default is state_data_extrap = false,
  // store_in_checkpoint = true.  We only need to put these in
  // explicitly if we want to do something different,
  // like not store the state data in a checkpoint directory
  bool state_data_extrap = false;
  bool store_in_checkpoint = true;

  amrex::BCRec bc;

  amrex::StateDescriptor::BndryFunc bndryfunc(nyx_bcfill);
  bndryfunc.setRunOnGPU(
      true); // I promise the bc function will launch gpu kernels.

  //////// Available Interpolators /////////////
  //         PCInterp                  pc_interp;
  //         NodeBilinear              node_bilinear_interp;
  //         FaceLinear                face_linear_interp;
  //         CellConservativeLinear    lincc_interp;
  //         CellConservativeLinear    cell_cons_interp(0);

  //         CellBilinear              cell_bilinear_interp;
  //         CellQuadratic             quadratic_interp;
  //         CellConservativeProtected protected_interp;
  //         CellConservativeQuartic   quartic_interp;
  //////////////////////////////////////////////

  Interpolater *interp;
  interp = &cell_cons_interp;

  // Establish the workhorse fields
  // std::cout << "Adding descriptors to desc_lst..." << std::endl;

  // desc_lst.addDescriptor(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                        amrex::IndexType::TheCellType(),
  //                        amrex::StateDescriptor::Point, 0, AxSCH::nFields(),
  //                        SCHinterp, state_data_extrap, store_in_checkpoint);
  //
  // desc_lst.addDescriptor(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                        amrex::IndexType::TheCellType(),
  //                        amrex::StateDescriptor::Point, 0, AxSCH::nFields(),
  //                        &cell_cons_interp, state_data_extrap,
  //                        store_in_checkpoint);

  desc_lst.addDescriptor(getState(StateType::PhiGrav_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 1, 1, &cell_cons_interp,
                         state_data_extrap, store_in_checkpoint);

  store_in_checkpoint = false;
  desc_lst.addDescriptor(
      getState(StateType::Gravity_Type), amrex::IndexType::TheCellType(),
      amrex::StateDescriptor::Point, 1, BL_SPACEDIM, &cell_cons_interp,
      state_data_extrap, store_in_checkpoint);

  // Set components
  // std::cout << "Setting components..." << std::endl;

  set_scalar_bc(bc, phys_bc);
  // desc_lst.setComponent(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                       AxSCH::getField(AxSCH::Fields::Dens), "Dens", bc,
  //                       bndryfunc);
  // desc_lst.setComponent(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                       AxSCH::getField(AxSCH::Fields::SCHf_Re), "SCHf_Re",
  //                       bc, bndryfunc);
  // desc_lst.setComponent(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                       AxSCH::getField(AxSCH::Fields::SCHf_Im), "SCHf_Im",
  //                       bc, bndryfunc);
  // desc_lst.setComponent(AxSCH::getState(AxSCH::StateType::SCH_Type),
  //                       AxSCH::getField(AxSCH::Fields::Phase), "SCHf_Im", bc,
  //                       bndryfunc);

  //
  // Print() << "getState(StateType::PhiGrav_Type) = "
  // << getState(StateType::PhiGrav_Type) << std::endl;
  // Print() << "get_field(Fields::PhiGrav) = " << getField(Fields::PhiGrav)
  // << std::endl;

  desc_lst.setComponent(getState(StateType::PhiGrav_Type),
                        getField(Fields::PhiGrav), "phi_grav", bc, bndryfunc);

  set_x_vel_bc(bc, phys_bc);
  desc_lst.setComponent(getState(StateType::Gravity_Type),
                        getField(Fields::GradPhi_X), "grav_x", bc, bndryfunc);
  set_y_vel_bc(bc, phys_bc);
  desc_lst.setComponent(getState(StateType::Gravity_Type),
                        getField(Fields::GradPhi_Y), "grav_y", bc, bndryfunc);
  set_z_vel_bc(bc, phys_bc);
  desc_lst.setComponent(getState(StateType::Gravity_Type),
                        getField(Fields::GradPhi_Z), "grav_z", bc, bndryfunc);
}
//
// Called when a *new* level is made (e.g., after regridding)
//
void AxSP::init() {
  AxSCHComov::init();
  // Get current time from previous level
  amrex::Real cur_time = static_cast<AxSP *>(&get_level(level - 1))
                             ->state[State_for_Time]
                             .curTime();

  // Initialize PhiGrav_Type (Gravitational Potential)
  amrex::MultiFab &phigrav_new =
      get_new_data(getState(StateType::PhiGrav_Type));
  FillCoarsePatch(phigrav_new, 0, cur_time, getState(StateType::PhiGrav_Type),
                  0, phigrav_new.nComp());

  // Initialize Gravity_Type (Gravitational Field)
  amrex::MultiFab &gradphi_new =
      get_new_data(getState(StateType::Gravity_Type));
  FillCoarsePatch(gradphi_new, 0, cur_time, getState(StateType::Gravity_Type),
                  0, gradphi_new.nComp());

  // Set dt to a large value to avoid affecting computeNewDt
  parent->setDtLevel(1.e100, level);
}
//
// Called when the level is initialized from a coarser, already existing level
//
void AxSP::init(amrex::AmrLevel &old) {
  AxSCHComov::init(old);
  // Retrieve old level data and current simulation time
  AxSP *old_level = static_cast<AxSP *>(&old);
  amrex::Real cur_time = old_level->state[State_for_Time].curTime();

  // Initialize PhiGrav_Type (Gravitational Potential)
  amrex::MultiFab &phigrav_new =
      get_new_data(getState(StateType::PhiGrav_Type));
  FillPatch(old, phigrav_new, 0, cur_time, getState(StateType::PhiGrav_Type), 0,
            1);

  // Initialize Gravity_Type (Gravitational Field)
  amrex::MultiFab &gradphi_new =
      get_new_data(getState(StateType::Gravity_Type));
  FillPatch(old, gradphi_new, 0, cur_time, getState(StateType::Gravity_Type), 0,
            BL_SPACEDIM);

  amrex::Gpu::Device::streamSynchronize();
}

//
// Initialize data on this level (including the wavefunction, etc.)
//
void AxSP::initData() {
  BL_PROFILE("AxSP::initData()");
  AxSCHComov::initData();

  // Initialize phi
  MultiFab &phigrav_new = get_new_data(getState(StateType::PhiGrav_Type));
  phigrav_new.setVal(0.);

  // Initialize gradphi
  MultiFab &gradphi_new = get_new_data(getState(StateType::Gravity_Type));
  gradphi_new.setVal(0.);

  if (!gravity) {
    amrex::Abort("Gravity object not initialized.");
  }

  gravity->set_mass_offset(0.0);

  int fill_interior = 0;
  int grav_n_grow = 1;
  gravity->solve_for_new_phi(level, phigrav_new,
                             gravity->get_grad_phi_curr(level), fill_interior,
                             grav_n_grow);

  // amrex::Print() << "checkpoint AxSP::initdata\n";
  // // #ifdef GRAV
  //
  // const int finest_level = parent->finestLevel();
  // BL_PROFILE_VAR("solve_for_old_phi", solve_for_old_phi);
  // // if (level == 0 || iteration > 1) {
  // MultiFab::RegionTag amrGrav_tag("Gravity_" + std::to_string(level));
  // // for (int lev = level; lev < finest_level; lev++) {
  // //   BaseGrav::gravity->zero_phi_flux_reg(lev + 1);
  // // }
  // // swap grav data
  // for (int lev = level; lev <= finest_level; lev++) {
  //   // get_level(lev).
  //   BaseGrav::gravity->swap_time_levels(lev);
  // }
  // // Solve for phi using the previous phi as a guess.
  // int use_previous_phi_as_guess = 1;
  // int ngrow_for_solve = 1; // iteration + stencil_deposition_width;
  // BaseGrav::gravity->multilevel_solve_for_old_phi(
  //     level, finest_level, ngrow_for_solve, use_previous_phi_as_guess);
  // // }
  // BL_PROFILE_VAR_STOP(solve_for_old_phi);
  // // #endif
}

//
// Compute the next time step, possibly factoring in comoving constraints
//
amrex::Real AxSP::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxSP::est_time_step()");

  amrex::Real cur_time = state[State_for_Time].curTime();
  // stop simulation if final_a is defined and reached:
  if (Comoving::final_a > 0.0) {
    amrex::Real current_a = Comoving::get_comoving_a(cur_time);
    if (current_a > Comoving::final_a) {
      parent->checkPoint();
      parent->writePlotFile();
      // Comoving::stop_at_final_a(Comoving::get_comoving_a(cur_time),
      //                           Comoving::final_a);
      amrex::Abort("AxSP:est_time_step() ::  a_now > a_final, Reached the end "
                   "condition");
    }
  }

  // In case we have a simple fixed time step.
  if (BaseAx::fixed_dt > 0)
    return BaseAx::fixed_dt;

  amrex::Real est_dt = 1.0e+200;

  if (vonNeumann_dt > 0) {

    amrex::Real a = Comoving::get_comoving_a(cur_time);
    // amrex::Real a = 1;
    const amrex::Real *dx = geom.CellSize();
    amrex::Real dt_cfl =
        dx[0] * dx[0] * a * a / 6.0 / hbaroverm; // stability condition for the
                                                 // SCH equation #ifdef GRAV
    const amrex::MultiFab &phi =
        get_new_data(getState(StateType::PhiGrav_Type));
    amrex::Real phi_max = std::abs(phi.max(0) - phi.min(0));
    dt_cfl = std::min(dt_cfl, hbaroverm / phi_max);

    // int nstep = parent->levelSteps(0) + 1;

    // if (dt_cfl == hbaroverm / phi_max) {
    //   Print() << "AxSP::est_time_step: Gravity is considered at: " << nstep
    //           << '\n';
    // }

    if ((level = 0 && PSorFD == 0)) // PSlevel
      dt_cfl *= vonNeumann_dt;
    else if (vonNeumann_dt < 1.0)
      dt_cfl *= vonNeumann_dt;

    est_dt = std::min(est_dt, dt_cfl);
    if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print() << "AxSP::est_time_step at level " << level
                     << ":  est_dt = " << est_dt << "\n";
    }
  }

  // Comoving est_time_step
  if (level == 0) {
    amrex::Real cur_time = state[State_for_Time].curTime();
    est_dt = Comoving::comoving_est_time_step(cur_time, est_dt);
  }
  // amrex::Print() << "AxSCHComov::comoving est_time_step at level " << level
  //                << ":  est_dt = " << est_dt << "\n";

  return est_dt;
}

void AxSP::average_down() {
  // Reuse AxSCHComov logic, or add extra comoving fields if you have them
  // AxSCHComov::average_down();
  if (level == parent->finestLevel())
    return;

  MultiFab &fine = get_new_data(AxSCH::getState(AxSCH::StateType::SCH_Type));
  MultiFab &coarse = get_level(level - 1).get_new_data(
      AxSCH::getState(AxSCH::StateType::SCH_Type));
  const IntVect &ratio = parent->refRatio(level);

  amrex::average_down(fine, coarse, 0, fine.nComp(), ratio);

  MultiFab &fine_phi = get_new_data(getState(StateType::PhiGrav_Type));
  MultiFab &coarse_phi =
      get_level(level - 1).get_new_data(getState(StateType::PhiGrav_Type));
  amrex::average_down(fine_phi, coarse_phi, 0, 1, ratio);

  MultiFab &fine_gradphi = get_new_data(getState(StateType::Gravity_Type));
  MultiFab &coarse_gradphi =
      get_level(level - 1).get_new_data(getState(StateType::Gravity_Type));
  amrex::average_down(fine_gradphi, coarse_gradphi, 0, 1, ratio);
}

int AxSP::nFields() {
  return AxSCH::nFields() + 4; // 4 new gravity fields
  // return 4; // 4 new gravity fields
}

int AxSP::getField(Fields f) {
  // int last_field = static_cast<int>(AxSCH::Fields::Phase);
  int last_field = 0;
  switch (f) {
  // case Fields::Dens:
  //   return 0;
  // case Fields::SCHf_Re:
  //   return 1;
  // case Fields::SCHf_Im:
  //   return 2;
  // case Fields::Phase:
  //   return 3;
  //
  case Fields::PhiGrav:
    return 0;
  case Fields::GradPhi_X:
    //
    return 0;
  case Fields::GradPhi_Y:
    return 1;
  case Fields::GradPhi_Z:
    return 2;
  }
  return -1;
}

AxSP::Fields AxSP::getField(int f) {
  int last_field = static_cast<int>(AxSCH::Fields::Phase);
  // == 3;
  switch (f) {
  // case 0:
  //   return Fields::Dens;
  // case 1:
  //   return Fields::SCHf_Re;
  // case 2:
  //   return Fields::SCHf_Im;
  // case 3:
  //   return Fields::Phase;
  //
  case 4:
    return Fields::PhiGrav;
    //
  case 5:
    return Fields::GradPhi_X;
  case 6:
    return Fields::GradPhi_Y;
  case 7:
    return Fields::GradPhi_Z;
  }
  return Fields::PhiGrav; // TODO: This should be an error value.
}

int AxSP::nStates() {
  // If AxSCHComov had 1 state (SCH_Type) = 0
  // we add 2 more here: PhiGrav_Type = 1, Gravity_Type = 2
  // => total = 3
  return 3;
}

int AxSP::getState(StateType st) {
  switch (st) {
    // NOTE: This must be the state used to track the time variable.
    //
  // case StateType::SCH_Type:
  //   return 0;
  case StateType::PhiGrav_Type:
    return 1;
  case StateType::Gravity_Type:
    return 2;
  }
  return -1;
}

AxSP::StateType AxSP::getState(int st) {
  switch (st) {
  // NOTE: This must be the state used to track the time variable.
  //
  // case 0:
  //   return StateType::SCH_Type;
  case 1:
    return StateType::PhiGrav_Type;
  case 2:
    return StateType::Gravity_Type;
  }
  return StateType::PhiGrav_Type; // TODO: This should be an error value.
}

// Retrieving the general density field and override it with initial density
// field from initData()
MultiFab &AxSP::get_density(bool old) {
  if (old) {
    // Print() << "AxSP::get_density: old" << '\n';
    return get_old_data(AxSCH::getState(AxSCH::StateType::SCH_Type));
  } else {
    // Print() << "AxSP::get_density: new" << '\n';
    return get_new_data(AxSCH::getState(AxSCH::StateType::SCH_Type));
  }
}

// computeNewDt is different to BaseAx, due to inclusion of comoving_a
void AxSP::computeNewDt(int finest_level, int sub_cycle,
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
    AxSP &adv_level = static_cast<AxSP &>(get_level(i));
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
void AxSP::computeInitialDt(int finest_level, int sub_cycle,
                            amrex::Vector<int> &n_cycle,
                            const amrex::Vector<amrex::IntVect> &ref_ratio,
                            amrex::Vector<amrex::Real> &dt_level,
                            amrex::Real stop_time) {
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
      AxSP &level_i = static_cast<AxSP &>(get_level(i));
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
      AxSP &level_i = static_cast<AxSP &>(get_level(i));
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
