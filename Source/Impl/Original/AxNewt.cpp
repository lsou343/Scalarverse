#include <AxNewt.H>
#include <NewtDerive.H>

#include <bc_fill.H>

#include <constants_cosmo.H>

#include <cmath>
#include <iostream>

using namespace amrex;

constexpr int Density_comp =
    0; // Define the density component if not included elsewhere

namespace {
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

AxNewt::AxNewt() {
  BL_PROFILE("AxNewt::AxNewt()");
  fine_mask = 0;
  std::cout << "AxNewt default constructor called." << std::endl;
}

AxNewt::AxNewt(Amr &papa, int lev, const Geometry &level_geom,
                       const BoxArray &bl, const DistributionMapping &dm,
                       Real time)
    : BaseNewt(papa, lev, level_geom, bl, dm, time,
               AxKG::getState(AxKG::StateType::KG_Type), AxKG::getState(AxKG::StateType::KG_Type),
               getState(StateType::PhiGrav_Type),
               getState(StateType::Gravity_Type)) {
  BL_PROFILE("AxNewt::AxNewt()");
  
  std::cout << "AxNewt constructor called for level " << lev << " at time "
            << time << "." << std::endl;

  if (level == 0 && time == 0.0) {
    std::cout << "Initializing time-dependent variables." << std::endl;
  }
}

AxNewt::~AxNewt() {
  std::cout << "AxNewt destructor called." << std::endl;

  // // Only delete `gravity` if this is the last level instance managing it
  // if (level == 0 && gravity && parent->finestLevel() == 0)
  // {
  //     delete gravity;
  //     gravity = nullptr;
  // }

  if (fine_mask) {
    delete fine_mask;
    fine_mask = nullptr;
  }
}

void AxNewt::init(AmrLevel &old) {
  AxKG::init(old);

  // Retrieve old level data and current simulation time
  AxNewt *old_level = static_cast<AxNewt *>(&old);
  amrex::Real cur_time = old_level->state[State_for_Time].curTime();

  // Initialize PhiGrav_Type (Gravitational Potential)
  amrex::MultiFab &phigrav_new = get_new_data(getState(StateType::PhiGrav_Type));
  FillPatch(old, phigrav_new, 0, cur_time, getState(StateType::PhiGrav_Type), 0, 1);

  // Initialize Gravity_Type (Gravitational Field)
  amrex::MultiFab &gradphi_new = get_new_data(getState(StateType::Gravity_Type));
  FillPatch(old, gradphi_new, 0, cur_time, getState(StateType::Gravity_Type), 0, BL_SPACEDIM);

  amrex::Gpu::Device::streamSynchronize();
}

void AxNewt::init() {
  AxKG::init();

  // Get current time from previous level
  amrex::Real cur_time = static_cast<AxNewt *>(&get_level(level - 1))	// LSR -- possibly redundant but keep for now
                             ->state[State_for_Time]
                             .curTime();

  // Initialize PhiGrav_Type (Gravitational Potential)
  amrex::MultiFab &phigrav_new = get_new_data(getState(StateType::PhiGrav_Type));
  FillCoarsePatch(phigrav_new, 0, cur_time, getState(StateType::PhiGrav_Type), 0, phigrav_new.nComp());

  // Initialize Gravity_Type (Gravitational Field)
  amrex::MultiFab &gradphi_new =
      get_new_data(getState(StateType::Gravity_Type));
  FillCoarsePatch(gradphi_new, 0, cur_time, getState(StateType::Gravity_Type), 0, gradphi_new.nComp());

  // Set dt to a large value to avoid affecting computeNewDt
  parent->setDtLevel(1.e100, level);
}

// Initialize all data
void AxNewt::initData() {
  printf("\n\n AxNewt::initData() \n\n");
#ifdef INFLATION
  AxKGComov::initData();
  printf("\n\n AxKGComov::initData() \n\n"); // LSR -- debug tool
#else
  printf("\n\n AxKG::initData() \n\n");
  AxKG::initData()
#endif
  printf("\n\nNext\n\n");
  BL_PROFILE("AxNewt::initData()");
  
  // Initialize phi
  MultiFab &phigrav_new = get_new_data(getState(StateType::PhiGrav_Type));
//  phigrav_new.setVal(0.); // LSR -- 0? Is this right? Commented out for now because I want phi to be nonzero

  std::cout << "AxNewt::initData: Checking phigrav_new allocation..."
            << std::endl;
  if (phigrav_new.nComp() <= 0 || phigrav_new.boxArray().size() == 0) {
    amrex::Abort("AxNewt::initData: state_new not properly initialized.");
  } else {
    std::cout << "AxNewt::initData: state_new has " << phigrav_new.nComp()
              << " components." << std::endl;
  }
  // TODO LSR is this for loop necessary or covered by AxKG?
  for (MFIter mfi(phigrav_new); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const &arr = phigrav_new.array(mfi);
    for (int k = bx.smallEnd(2); k <= bx.bigEnd(2); ++k) {
      for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
        for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
          try {
            prob_initdata_pos(i, j, k, arr, geom.data(), {});
          } catch (const std::exception &e) {
            amrex::Abort("Exception caught in initData.");
          }
        }
      }
    }
  }

  // Initialize gradphi
  MultiFab &gradphi_new = get_new_data(getState(StateType::Gravity_Type));
  gradphi_new.setVal(0.);

  if (!gravity) {
    amrex::Abort("Gravity object not initialized.");
  }

  gravity->set_mass_offset(0.0);

  amrex::Print() << "checkpoint AxNewt::initdate\n";
}

amrex::Real AxNewt::advance(amrex::Real time, amrex::Real dt_old,
                                int iteration, int ncycle) {
  BL_PROFILE("AxNewt::advance()");

  amrex::Real dt = est_time_step(dt_old);

  // Print diagnostic information
  amrex::Print() << "AxNewt::advance at time " << time << " with dt " << dt
                 << std::endl;

  // Get the MultiFab for gravitational potential (PhiGrav_Type)
  MultiFab &phi_new = get_new_data(getState(StateType::PhiGrav_Type));
  // Get the cell-centered MultiFab for the gravitational field (Gravity_Type)
  MultiFab &grav_new = get_new_data(getState(StateType::Gravity_Type));

  // Check that the Gravity object is initialized
  if (BaseNewt::gravity == nullptr) {
    amrex::Abort("Gravity object is not initialized in AxNewt::advance");
  }

  // Set parameters for the solver (adjust these as needed)
  int fill_interior = 1;
  int ngrow_for_solve = 1;
  int lev = level; // current AMR level

  // Create temporary edge-centered MultiFabs for each spatial direction to hold
  // the gradient of phi.
  Vector<MultiFab *> grad_phi;
  for (int d = 0; d < BL_SPACEDIM; ++d) {
    MultiFab *mf = new MultiFab(getEdgeBoxArray(d), DistributionMap(), 1, 1);
    mf->setVal(0.0);
    grad_phi.push_back(mf);
  }

  // Call the gravity solver to compute the new gravitational potential.
  // This call uses the current density field (already set up) to compute
  // phi_new and its edge gradients.
  BaseNewt::gravity->solve_for_new_phi(lev, phi_new, grad_phi, fill_interior,
                                       ngrow_for_solve);

  // Now average the computed edge gradients to obtain a cell-centered
  // gravitational field.
  Vector<const MultiFab *> grad_phi_const;
  for (int d = 0; d < BL_SPACEDIM; ++d) {
    grad_phi_const.push_back(grad_phi[d]);
  }
  amrex::average_face_to_cellcenter(grav_new, grad_phi_const, geom);

  // Clean up temporary edge data.
  for (int d = 0; d < BL_SPACEDIM; ++d) {
    delete grad_phi[d];
  }
  grad_phi.clear();

  // Update the time level (this example simply advances by dt; adjust if you
  // compute a new dt)
  setTimeLevel(time + dt, dt, dt);

  // Diagnostics for φ (gravitational potential)
  amrex::Real phi_norm = phi_new.norm2();
  amrex::Print() << "Diagnostics for φ:" << "\n"
                 << "  L₂ norm  = " << phi_norm << "\n";
  return dt;
}

// probably not needed?
void AxNewt::prob_initdata_mom(
    int i, int j, int k,
    const amrex::Array4<amrex::GpuComplex<amrex::Real>> &fields, // LSR -- switched word order so now const amrex... rather than amrex... const
    const amrex::GeometryData &geomdata,
    const amrex::GpuArray<double, 10> &prob_param) {
  const Real *dx = geomdata.CellSize();

  // Ensure we do not go out of bounds
  const int im1 = (i > 0) ? i - 1 : i;  // LSR -- Doesn't this double up on the boundaries? Should loop back around I think. Or is it a way to ignore boundaries to do them later?
  const int ip1 = (i < geomdata.Domain().bigEnd(0)) ? i + 1 : i;
  const int jm1 = (j > 0) ? j - 1 : j;
  const int jp1 = (j < geomdata.Domain().bigEnd(1)) ? j + 1 : j;
  const int km1 = (k > 0) ? k - 1 : k;
  const int kp1 = (k < geomdata.Domain().bigEnd(2)) ? k + 1 : k;
//  const int im1 = (i > 0) ? i - 1 : geomdata.Domain().bigEnd(0);  // LSR -- Unsure if this is the correct way to go about this re above, need to consider. Doesn't fix the problem anyway (:
//  const int ip1 = (i < geomdata.Domain().bigEnd(0)) ? i + 1 : 0;
//  const int jm1 = (j > 0) ? j - 1 : geomdata.Domain().bigEnd(1);
//  const int jp1 = (j < geomdata.Domain().bigEnd(1)) ? j + 1 : 0;
//  const int km1 = (k > 0) ? k - 1 : geomdata.Domain().bigEnd(2);
//  const int kp1 = (k < geomdata.Domain().bigEnd(2)) ? k + 1 : 0;
//  printf("\n\n%i, %i, %i\n\n", i, j, k); // LSR -- Debug
  // Compute finite difference approximation for gravitational field (grad phi)
  fields(i, j, k, getField(Fields::GradPhi_X)) =
      (fields(ip1, j, k, getField(Fields::PhiGrav)) -
       fields(im1, j, k, getField(Fields::PhiGrav))) /
      (2.0 * dx[0]);

  fields(i, j, k, getField(Fields::GradPhi_Y)) =
      (fields(i, jp1, k, getField(Fields::PhiGrav)) -
       fields(i, jm1, k, getField(Fields::PhiGrav))) /
      (2.0 * dx[1]);

  fields(i, j, k, getField(Fields::GradPhi_Z)) =
      (fields(i, j, kp1, getField(Fields::PhiGrav)) -
       fields(i, j, km1, getField(Fields::PhiGrav))) /
      (2.0 * dx[2]);
  // Debug Output
  if (i == 0 && j == 0 && k == 0) { // Print for one cell only
    amrex::Print() << "Gravitational field initialized at (0,0,0): "
                   << "gradPhiX="
                   << fields(i, j, k, getField(Fields::GradPhi_X)) << ", "
                   << "gradPhiY="
                   << fields(i, j, k, getField(Fields::GradPhi_Y)) << ", "
                   << "gradPhiZ="
                   << fields(i, j, k, getField(Fields::GradPhi_Z)) << "\n"
                   << "Phi_grav = "						// LSR -- just checking
                   << fields(i, j, k, getField(Fields::PhiGrav)) << "\n"; 	// LSR -- just checking
  }
}

amrex::Real AxNewt::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxNewt::est_time_step()");

  // Currently the only option
  if (BaseAx::fixed_dt > 0)
    return BaseAx::fixed_dt;

  return 0;

  //     const Real safety_factor = 0.5;  // A safety factor to prevent
  //     numerical instability const Real Ggravity = 1.0;       // Ensure this
  //     is correctly defined elsewhere

  //     // Debug: Check if Ggravity is correctly initialized
  //     amrex::Print() << "Ggravity: " << Ggravity << std::endl;

  //     MultiFab& density_mf = get_new_data(getState(StateType::State_Type));

  //     // Compute max density in the domain
  //     Real max_density = 0.0;
  //     for (MFIter mfi(density_mf); mfi.isValid(); ++mfi) {
  //         max_density = std::max(max_density, density_mf[mfi].max(0));
  //     }
  //     ParallelDescriptor::ReduceRealMax(max_density);

  //     // Debug: Check if density is properly initialized
  //     amrex::Print() << "Max density in domain: " << max_density <<
  //     std::endl;

  //     // Compute free-fall time step
  //     Real dt_grav = std::numeric_limits<Real>::max();
  //     if (max_density > 0.0) {
  //         dt_grav = safety_factor / std::sqrt(Ggravity * max_density);
  //     } else {
  //         amrex::Print() << "Warning: max_density is 0. Check
  //         initialization!" << std::endl;
  //     }

  //     // Debug: Print computed dt_grav
  //     amrex::Print() << "Computed dt_grav: " << dt_grav << std::endl;

  //     // Ensure dt is not zero
  //     Real min_dt = 1e-3;
  //     Real new_dt = std::max(std::min(dt_old, dt_grav), min_dt);

  //     amrex::Print() << "Estimated dt (gravity-based): " << new_dt <<
  //     std::endl; return new_dt;
}

// Average data from finer to coarser levels
void AxNewt::average_down() {
#ifdef INFLATION
  AxKGComov::average_down();  // LSR -- BaseAx -> AxKG
#else
  AxKG::average_down()
#endif
  if (level == parent->finestLevel())
    return;
  // LSR -- TODO Figure out whatever is going on here. Do we need to add something for derivative? Also why is there no average_down for field values in AxKG?
  MultiFab &fine_phi = get_new_data(getState(StateType::PhiGrav_Type));
  MultiFab &coarse_phi =
      get_level(level - 1).get_new_data(getState(StateType::PhiGrav_Type));
      
  const IntVect &ratio = parent->refRatio(level);
  
  amrex::average_down(fine_phi, coarse_phi, 0, 1, ratio);

  MultiFab &fine_gradphi = get_new_data(getState(StateType::Gravity_Type));
  MultiFab &coarse_gradphi =
      get_level(level - 1).get_new_data(getState(StateType::Gravity_Type));
  amrex::average_down(fine_gradphi, coarse_gradphi, 0, 1, ratio);
}

// Add the variables to do with gravity. This will also call the Klein-Gordon solvers. Note that this does not initialise anything.
// TODO: Incorporate Schroedinger solvers here too - one gravitational implementation is probably sufficient :)
void AxNewt::variable_setup() {
  printf("\n\nAxNewt::variable_setup()\n\n");
//#ifdef INFLATION
//  AxKGComov::variable_setup();  // LSR -- Currently does nothing but redirect to AxKG, but worthwhile in case that changes
//#else
//  AxKG::variable_setup();
//#endif
  
  bool state_data_extrap = false;
  bool store_in_checkpoint = true;
  
  amrex::BCRec bc;
  
  amrex::StateDescriptor::BndryFunc bndryfunc(nyx_bcfill);
  bndryfunc.setRunOnGPU(true);
    
  Interpolater *interp;
  interp = &cell_cons_interp;

  // Establish the additional workhorse fields
  std::cout << "Adding descriptors to desc_lst..." << std::endl;

  desc_lst.addDescriptor(getState(StateType::PhiGrav_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, /* nextra? 0 for KGf and KGfv */ 1, /* number of fields */ nFields(), &cell_cons_interp,
                         state_data_extrap, store_in_checkpoint);

  store_in_checkpoint = false;	// LSR -- does this need to be part of the multifab?
  desc_lst.addDescriptor(
      getState(StateType::Gravity_Type), amrex::IndexType::TheCellType(),
      amrex::StateDescriptor::Point, 1, BL_SPACEDIM, &cell_cons_interp,
      state_data_extrap, store_in_checkpoint);

  // Set components
  std::cout << "Setting components..." << std::endl;
  set_scalar_bc(bc, phys_bc);
  desc_lst.setComponent(getState(StateType::Density),
                        getField(Fields::Density), "density", bc, bndryfunc);
  desc_lst.setComponent(getState(StateType::PhiGrav_Type),
                        getField(Fields::PhiGrav), "phi_grav", bc, bndryfunc);
  desc_lst.setComponent(getState(StateType::PhiGrav_Type),
                        getField(Fields::PhiGravv), "phi_gravv", bc, bndryfunc);

//  set_x_vel_bc(bc, phys_bc); // LSR -- What's going on here? What are the boundary conditions?
//  desc_lst.setComponent(getState(StateType::Gravity_Type),
//                        getField(Fields::GradPhi_X), "grav_x", bc, bndryfunc);
//  set_y_vel_bc(bc, phys_bc);
//  desc_lst.setComponent(getState(StateType::Gravity_Type),
//                        getField(Fields::GradPhi_Y), "grav_y", bc, bndryfunc);
//  set_z_vel_bc(bc, phys_bc);
//  desc_lst.setComponent(getState(StateType::Gravity_Type),
//                        getField(Fields::GradPhi_Z), "grav_z", bc, bndryfunc);

  std::cout << "Variable setup complete." << std::endl;
  printf("\n\nAxNewt::variable_setup done\n\n");
}

// Helper functions to map fields and states. This will be very useful when
// combining different types of simulations (e.g., KG, gravity, particles, etc.)
int AxNewt::nFields() {
  // We have two fields right now,
  // KGf - inflaton field
  // KGfv - inflaton field derivative
  // Density - energy density (derived but needed at every step)
  // PhiGrav - Newtonian potential
  // PhiGravv - Newtonian potential derivative
  return 3;
}

int AxNewt::getField(Fields f) {
  int field = -1;
  switch (f) {
//case Fields::KGf:
//  field = 0;
//case Fields::KGfv:
//  field = 1;
  case Fields::Density:
    field = 2;
    break;
  case Fields::PhiGrav:
    field = 3;
    break;
  case Fields::PhiGravv:	// LSR -- oh no I don't like the double v. TODO: find a better name!
    field = 4;
    break;
//  case Fields::GradPhi_X:
//    field = 0;
//    break;
//  case Fields::GradPhi_Y:
//    field = 1;
//    break;
//  case Fields::GradPhi_Z:
//    field = 2;
//    break;
  }
  if (field == -1) {
    std::cerr << "Invalid field requested!" << std::endl;
    amrex::Abort("Invalid field.");
  }
  return field;
}

int AxNewt::nStates() { return 3; }

int AxNewt::getState(StateType st) {
  int state = -1;
  switch (st) {
  case StateType::Density_Type:
    state = 1;
    break;
  case StateType::PhiGrav_Type:
    state = 2;
    break;
  case StateType::PhiGravv_Type:
    state = 3;
    break;
  }
  if (state == -1) {
    std::cerr << "Invalid state requested!" << std::endl;
    amrex::Abort("Invalid state.");
  }
  return state;
}
