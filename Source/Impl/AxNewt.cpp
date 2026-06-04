#include <AxNewt.H>
#include <NewtDerive.H>

#include <bc_fill.H>

#include <constants_cosmo.H>

#include <cmath>
#include <iostream>
// LSR -- TODO: Once density is working, rename Dens for consistency with SCH
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
               getState(StateType::PhiGrav_Type)) {
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
//
// Called when the level is initialized from a coarser, already existing level
//
void AxNewt::init(AmrLevel &old) {
  AxKG::init(old);

  amrex::MultiFab&  density_new = get_new_data(getState(StateType::Density_Type));

  AxNewt* old_level = static_cast<AxNewt*> (&old);
  amrex::Real cur_time  = old_level->state[State_for_Time].curTime();

  FillPatch(old, density_new, 0, cur_time, getState(StateType::Density_Type), 0, nFields());

  amrex::Gpu::Device::streamSynchronize();

}
//
// Called when a *new* level is made (e.g., after regridding)
//
void AxNewt::init() {
  AxKG::init();

  amrex::Real cur_time  = static_cast<AxNewt*>(&get_level(level-1))->state[State_for_Time].curTime();

  amrex::MultiFab&  Dens_new = get_new_data(getState(StateType::Density_Type));
  FillCoarsePatch(Dens_new, 0, cur_time, getState(StateType::Density_Type), 0, Dens_new.nComp());
    
  // We set dt to be large for this new level to avoid screwing up
  // computeNewDt.
  parent->setDtLevel(1.e100, level);
}

// Initialize all data
void AxNewt::initData() {
  BL_PROFILE("AxNewt::initData()");
  printf("\n\n AxNewt::initData() \n\n");
#ifdef INFLATION
  AxKGComov::initData();
  printf("\n\n AxKGComov::initData() \n\n"); // LSR -- debug tool
#else
  printf("\n\n AxKG::initData() \n\n");
  AxKG::initData()
#endif
  
  // LSR -- So now the field and field derivative are ready, next step is add density... how? Schroedinger defines it initially and then finds its value in timestep, is this the way forward?

  // Regardless of pos or mom, we have pos field values now
  // 1. Find how to call them
  // 2. Set density using phidot^2 + grad^2 phi + V(phi)
  // 3. Caclulate initial Phi value
  // 4. Set Phidot = 0

  amrex::MultiFab& density_new = get_density();  // LSR -- get_density returns zero at all points on the grid which is obviously not ideal. Need to figure out a way to have it accurately find edens
  amrex::MultiFab&  KG_new = get_level(level).get_new_data(AxKG::getState(AxKG::StateType::KG_Type));  // LSR -- TODO: figure out if I need new or old. Think it's new but double check - no need to recalculate values if that is what new does

  // ALSO: may not work generally because it relies too heavily on KGComov - need a solution that is independent of KGComov
  
  const amrex::Real invdeltsq = 1.0 / geom.CellSizeArray[0] / geom.CellSizeArray[0]; // NEW TODO: geom instead of geomdata. Also see if just dx from AxKG works (suspect no)
  
  amrex::Real tmp_grad = 0., tmp_pot = 0., tmp_kin = 0.;
  for (amrex::MFIter mfi(KG_new, false); mfi.isValid(); ++mfi) {
    amrex::Array4<amrex::Real> const &arr = KG_new.array(mfi);
    const amrex::Box &bx = mfi.validbox();

    amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) {
      
      // tmp_pot = Models::compute_model_quantity(arr, comp, a, ap, app, quantity) // NEW TODO: Figure this out
      tmp_kin = 1.;

    });
    // NEW TODO: In BaseAx.H, what is a) cMultifab, b) FillK, c) fab_new, and how can I use equivalents here?
    // Note: I'm going to keep everything in here for now but eventually I will create init_density() and init_phi() functions in Newtonian.H or Newtonian.cpp
  }
  Comoving::add_to_rho(tmp_grad, tmp_pot, tmp_kin); // This doesn't work the way I hoped
  
  // Follow up: I'm setting density_new to be the density but is that actually getting saved to the output? Probably not
  
}

amrex::Real AxNewt::advance(amrex::Real time, amrex::Real dt_old,
                                int iteration, int ncycle) {
  BL_PROFILE("AxNewt::advance()");

  amrex::Real dt = est_time_step(dt_old);

  // Print diagnostic information
  amrex::Print() << "AxNewt::advance at time " << time << " with dt " << dt
                 << std::endl;

  return dt;
}


amrex::Real AxNewt::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxNewt::est_time_step()");

  // Currently the only option
  if (BaseAx::fixed_dt > 0)
    return BaseAx::fixed_dt;

  return 0;
}

// Average data from finer to coarser levels
void AxNewt::average_down() {
  BL_PROFILE("AxNewt::average_down()");
#ifdef INFLATION
  AxKGComov::average_down();  // LSR -- BaseAx -> AxKG
#else
  AxKG::average_down()
#endif
  
  if (level == parent->finestLevel())
    return;

  BaseAx::average_down(getState(StateType::Density_Type));
  // LSR -- This shouldn't be getting called at the moment since we are only using one level
}

// Add the variables to do with gravity. This will also call the Klein-Gordon solvers. Note that this does not initialise anything.
// LSR -- TODO: Incorporate Schroedinger solvers here too - one gravitational implementation is probably sufficient :)
void AxNewt::variable_setup() {
  printf("\n\nAxNewt::variable_setup()\n\n");
#ifdef INFLATION
  AxKGComov::variable_setup();  // LSR -- Currently does nothing but redirect to AxKG, but worthwhile in case that changes
#else
  AxKG::variable_setup();
#endif
  AxKG::read_params();
  
  bool state_data_extrap = false;
  bool store_in_checkpoint = true;

  amrex::BCRec bc;

  amrex::StateDescriptor::BndryFunc bndryfunc(nyx_bcfill);
  bndryfunc.setRunOnGPU(true);

  Interpolater *interp;
  interp = &cell_cons_interp;

  // Establish the additional workhorse fields
  std::cout << "Adding descriptors to desc_lst..." << std::endl;

  desc_lst.addDescriptor(getState(StateType::Density_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 1, 1 /* nfields() */, &cell_cons_interp,
                         state_data_extrap, store_in_checkpoint);

  // Set components
  std::cout << "Setting components..." << std::endl;
  set_scalar_bc(bc, phys_bc);
  
  desc_lst.setComponent(getState(StateType::Density_Type),
                        getField(Fields::Density), "density", bc, bndryfunc);

  // LSR -- TODO: add gravity stuff

}

// Helper functions to map fields and states. This will be very useful when
// combining different types of simulations (e.g., KG, gravity, particles, etc.)
int AxNewt::nFields() {
  // We have two fields from AxKG (KGf and KGfv)
  // We also have two gravity fields
  // PhiGrav - Newtonian potential
  // PhiGravv - Newtonian potential derivative
  return 2;
}

int AxNewt::getField(Fields f) {
  int field = -1;
  switch (f) {
//case Fields::KGf:
//  field = 0;
//case Fields::KGfv:
//  field = 1;
  case Fields::Density:
    field = 0;
    break;
//case Fields::PhiGrav:
//  field = 0;
//case Fields::PhiGravv:
//  field = 1;
  }
  if (field == -1) {
    std::cerr << "Invalid field requested!" << std::endl;
    amrex::Abort("Invalid field.");
  }
  return field;
}

int AxNewt::nStates() { return 2; }

int AxNewt::getState(StateType st) {
  int state = -1;
  switch (st) {
  case StateType::Density_Type:
    state = 1;
    break;
  case StateType::PhiGrav_Type:
    state = 2;
    break;
//  case StateType::PhiGravv_Type: // LSR -- might not actually want this, instead fold into PhiGrav_Type as an extra component
//    state = 3;
//    break;
  }
  if (state == -1) {
    std::cerr << "Invalid state requested!" << std::endl;
    amrex::Abort("Invalid state.");
  }
  return state;
}

// Retrieving the general density field and override it with initial density
// field from initData() - LSR -- Why? Why not fold this into init?
// LSR -- also think this won't work - currently no data to get. Need somewhere where density is being calculated and updated.
MultiFab &AxNewt::get_density(bool old) {
  if (old) {
    return get_old_data(AxNewt::getState(AxNewt::StateType::Density_Type));
  } else {
    return get_new_data(AxNewt::getState(AxNewt::StateType::Density_Type));
  }
}
