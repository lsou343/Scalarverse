#include <AxTestGrav.H>
#include <TestGravDerive.H>

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

AxTestGrav::AxTestGrav() {
  BL_PROFILE("AxTestGrav::AxTestGrav()");
  fine_mask = 0;
  std::cout << "AxTestGrav default constructor called." << std::endl;
}

AxTestGrav::AxTestGrav(Amr &papa, int lev, const Geometry &level_geom,
                       const BoxArray &ba, const DistributionMapping &dm,
                       Real time)
    : BaseGrav(papa, lev, level_geom, ba, dm, time,
               getState(StateType::State_Type), getState(StateType::State_Type),
               getState(StateType::PhiGrav_Type),
               getState(StateType::Gravity_Type)) {
  BL_PROFILE("AxTestGrav::AxTestGrav()");

  std::cout << "AxTestGrav constructor called for level " << lev << " at time "
            << time << "." << std::endl;

  if (level == 0 && time == 0.0) {
    std::cout << "Initializing time-dependent variables." << std::endl;
  }
}

AxTestGrav::~AxTestGrav() {
  std::cout << "AxTestGrav destructor called." << std::endl;

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

void AxTestGrav::init(AmrLevel &old) {
  BaseAx::init(old);

  // Retrieve old level data and current simulation time
  AxTestGrav *old_level = static_cast<AxTestGrav *>(&old);
  amrex::Real cur_time = old_level->state[State_for_Time].curTime();

  // Initialize State_Type
  amrex::MultiFab &state_new = get_new_data(getState(StateType::State_Type));
  FillPatch(old, state_new, 0, cur_time, getState(StateType::State_Type), 0, 1);

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

void AxTestGrav::init() {
  BaseAx::init();

  // Get current time from previous level
  amrex::Real cur_time = static_cast<AxTestGrav *>(&get_level(level - 1))
                             ->state[State_for_Time]
                             .curTime();

  // Initialize State_Type
  amrex::MultiFab &state_new = get_new_data(getState(StateType::State_Type));
  FillCoarsePatch(state_new, 0, cur_time, getState(StateType::State_Type), 0,
                  state_new.nComp());

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

// Initialize all data
void AxTestGrav::initData() {
  BL_PROFILE("AxTestGrav::initData()");

  // Initialize density distribution
  MultiFab &state_new = get_new_data(getState(StateType::State_Type));

  std::cout << "AxTestGrav::initData: Checking state_new allocation..."
            << std::endl;
  if (state_new.nComp() <= 0 || state_new.boxArray().size() == 0) {
    amrex::Abort("AxTestGrav::initData: state_new not properly initialized.");
  } else {
    std::cout << "AxTestGrav::initData: state_new has " << state_new.nComp()
              << " components." << std::endl;
  }

  for (MFIter mfi(state_new); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const &arr = state_new.array(mfi);
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

  amrex::Print() << "checkpoint AxTestGrav::initdate\n";
}

// Retrieving the general density field and override it with initial density
// field from initData()
MultiFab &AxTestGrav::get_density(bool old) {
  if (old) {

    return get_old_data(getState(StateType::State_Type));
  } else {
    return get_new_data(getState(StateType::State_Type));
  }
}

amrex::Real AxTestGrav::advance(amrex::Real time, amrex::Real dt_old,
                                int iteration, int ncycle) {
  BL_PROFILE("AxTestGrav::advance()");

  amrex::Real dt = est_time_step(dt_old);

  // Print diagnostic information
  amrex::Print() << "AxTestGrav::advance at time " << time << " with dt " << dt
                 << std::endl;

  // Get the MultiFab for gravitational potential (PhiGrav_Type)
  MultiFab &phi_new = get_new_data(getState(StateType::PhiGrav_Type));
  // Get the cell-centered MultiFab for the gravitational field (Gravity_Type)
  MultiFab &grav_new = get_new_data(getState(StateType::Gravity_Type));

  // Check that the Gravity object is initialized
  if (BaseGrav::gravity == nullptr) {
    amrex::Abort("Gravity object is not initialized in AxTestGrav::advance");
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
  BaseGrav::gravity->solve_for_new_phi(lev, phi_new, grad_phi, fill_interior,
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

// Initialize the density field
void AxTestGrav::prob_initdata_pos(
    int i, int j, int k, const amrex::Array4<double> &fields,
    const amrex::GeometryData &geomdata,
    const amrex::GpuArray<double, 10> &prob_param) {
  const Real *prob_lo = geomdata.ProbLo();
  const Real *dx = geomdata.CellSize();

  // Compute grid coordinates
  Real x = prob_lo[0] + i * dx[0];
  Real y = prob_lo[1] + j * dx[1];
  Real z = prob_lo[2] + k * dx[2];

  // Define sphere parameters (centered at origin)
  Real center[3] = {100.0, 100.0, 100.0};
  Real radius = 5;
  Real dist_sq = (x - center[0]) * (x - center[0]) +
                 (y - center[1]) * (y - center[1]) +
                 (z - center[2]) * (z - center[2]);
  // Real dist_sq = x * x + y * y + z * z;  // Optimized calculation

  // Assign density based on sphere condition
  Real density_value = (dist_sq <= radius * radius) ? 5.0e40 : 0.0;

  try {
    fields(i, j, k, AxTestGrav::getField(Fields::Density)) = density_value;
  } catch (const std::exception &e) {
    std::cerr << "Error during density field initialization: " << e.what()
              << std::endl;
    amrex::Abort("Exception in prob_initdata_pos.");
  }

  fields(i, j, k, AxTestGrav::getField(Fields::Density)) = density_value;
  // std::cout << "Density initialized at position (" << x << ", " << y << ", "
  // << z << ") with value " << density_value << "." << std::endl;
}

// probably not needed?
void AxTestGrav::prob_initdata_mom(
    int i, int j, int k,
    amrex::Array4<amrex::GpuComplex<amrex::Real>> const &fields,
    const amrex::GeometryData &geomdata,
    const amrex::GpuArray<double, 10> &prob_param) {
  printf("\n\n Hello \n\n");
  const Real *dx = geomdata.CellSize();

  // Ensure we do not go out of bounds
  const int im1 = (i > 0) ? i - 1 : i;
  const int ip1 = (i < geomdata.Domain().bigEnd(0)) ? i + 1 : i;
  const int jm1 = (j > 0) ? j - 1 : j;
  const int jp1 = (j < geomdata.Domain().bigEnd(1)) ? j + 1 : j;
  const int km1 = (k > 0) ? k - 1 : k;
  const int kp1 = (k < geomdata.Domain().bigEnd(2)) ? k + 1 : k;

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
  print("\n\n %i %i %i \n\n", i, j, k);
  if (i == 0 && j == 0 && k == 0) { // Print for one cell only
    amrex::Print() << "Gravitational field initialized at (0,0,0): "
                   << "gradPhiX="
                   << fields(i, j, k, getField(Fields::GradPhi_X)) << ", "
                   << "gradPhiY="
                   << fields(i, j, k, getField(Fields::GradPhi_Y)) << ", "
                   << "gradPhiZ="
                   << fields(i, j, k, getField(Fields::GradPhi_Z)) << "\n";
  }
}

amrex::Real AxTestGrav::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxTestGrav::est_time_step()");

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
void AxTestGrav::average_down() {
  if (level == parent->finestLevel())
    return;

  MultiFab &fine = get_new_data(getState(StateType::State_Type));
  MultiFab &coarse =
      get_level(level - 1).get_new_data(getState(StateType::State_Type));
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

void AxTestGrav::variable_setup() {
  BaseAx::variable_setup();

  // Get options, set phys_bc  --- This is necessary because these are all
  // static functions, so there's no actual inheritance!
  AxTestGrav::read_params();

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
  std::cout << "Adding descriptors to desc_lst..." << std::endl;

  desc_lst.addDescriptor(getState(StateType::State_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 0, 1, &cell_cons_interp,
                         state_data_extrap, store_in_checkpoint);

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
  std::cout << "Setting components..." << std::endl;
  set_scalar_bc(bc, phys_bc);
  desc_lst.setComponent(getState(StateType::State_Type),
                        getField(Fields::Density), "density", bc, bndryfunc);
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

  // Establish the derived fields
  // Log(density)
  // derive_lst.add("logden", IndexType::TheCellType(), 1, Derived::derlogden,
  // Derived::the_same_box); derive_lst.addComponent("logden", desc_lst,
  // State_Type, Density_comp, 1);

  // derive_lst.add("StateErr", IndexType::TheCellType(), 3, Derived::derstate,
  // Derived::grow_box_by_one); derive_lst.addComponent("StateErr", desc_lst,
  // State_Type, Density_comp, 1);
  //

  std::cout << "Variable setup complete." << std::endl;
}

// Helper functions to map fields and states. This will be very useful when
// combining different types of simulations (e.g., KG, gravity, particles, etc.)
int AxTestGrav::nFields() {
  // We have three fields right now,
  // State_Type - density
  // PhiGrav_Type - phi (gravitational potential)
  // Gravity_Type - grad_phi (gravitational field)
  return 3;
}

int AxTestGrav::getField(Fields f) {
  int field = -1;
  switch (f) {
  case Fields::Density:
    field = 0;
    break;
  case Fields::PhiGrav:
    field = 0;
    break;
  case Fields::GradPhi_X:
    field = 0;
    break;
  case Fields::GradPhi_Y:
    field = 1;
    break;
  case Fields::GradPhi_Z:
    field = 2;
    break;
  }
  if (field == -1) {
    std::cerr << "Invalid field requested!" << std::endl;
    amrex::Abort("Invalid field.");
  }
  return field;
}

int AxTestGrav::nStates() { return 3; }

int AxTestGrav::getState(StateType st) {
  int state = -1;
  switch (st) {
  case StateType::State_Type:
    state = 0;
    break;
  case StateType::PhiGrav_Type:
    state = 1;
    break;
  case StateType::Gravity_Type:
    state = 2;
    break;
  }
  if (state == -1) {
    std::cerr << "Invalid state requested!" << std::endl;
    amrex::Abort("Invalid state.");
  }
  return state;
}
