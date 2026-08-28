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
  printf("\n\ninit(old)\n\n");

  amrex::MultiFab&  density_new = get_new_data(getState(StateType::Density_Type));
  amrex::MultiFab&  Phi_new = get_new_data(getState(StateType::PhiGrav_Type));

  AxNewt* old_level = static_cast<AxNewt*> (&old);
  amrex::Real cur_time  = old_level->state[State_for_Time].curTime();

  FillPatch(old, density_new, 0, cur_time, getState(StateType::Density_Type), 0, 1);
  FillPatch(old, Phi_new, 0, cur_time, getState(StateType::PhiGrav_Type), 0, nFields());

  amrex::Gpu::Device::streamSynchronize();

}
//
// Called when a *new* level is made (e.g., after regridding)
//
void AxNewt::init() {
  AxKG::init();
  printf("\n\ninit\n\n");

  amrex::Real cur_time  = static_cast<AxNewt*>(&get_level(level-1))->state[State_for_Time].curTime();

  amrex::MultiFab&  Dens_new = get_new_data(getState(StateType::Density_Type));
  FillCoarsePatch(Dens_new, 0, cur_time, getState(StateType::Density_Type), 0, Dens_new.nComp());

  amrex::MultiFab&  Phi_new = get_new_data(getState(StateType::PhiGrav_Type));
  FillCoarsePatch(Phi_new, 0, cur_time, getState(StateType::PhiGrav_Type), 0, Phi_new.nComp());

  // We set dt to be large for this new level to avoid screwing up
  // computeNewDt.
  parent->setDtLevel(1.e100, level);
}

// Initialize all data
void AxNewt::initData() {
  BL_PROFILE("AxNewt::initData()");
#ifdef INFLATION
  AxKGComov::initData();
  amrex::Real a = Comoving::get_comoving_a(),
              ap = Comoving::get_comoving_ap();
#else
  AxKG::initData()
  amrex::Real a = 1.,	// If the universe is not expanding, take the scale factor and its derivative to be 1 and 0
              ap = 0.;
#endif
  printf("\n\ninitData\n\n");

  if (!gravity) {
    amrex::Abort("Gravity object not initialized.");
  }

  amrex::BoxArray ba;
  amrex::DistributionMapping dm;

  // Initialise the field multifabs
  amrex::MultiFab& KG_new = get_new_data(AxKG::getState(AxKG::StateType::KG_Type));  // LSR -- get_level(level).get_new_data -> get_new_data
  amrex::MultiFab& density_new = get_density();
  amrex::MultiFab& PhiGrav_new = get_new_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));
  PhiGrav_new.setVal(0.);

  KG_new.FillBoundary(geom.periodicity());

  // Define some useful constants
  const amrex::Real *dx = geom.CellSize();
  const amrex::Real invdeltsq = 1.0 / dx[0] / dx[0];  // LSR -- what happens with mesh refinement here?

  gravity->solve_density_data(level, KG_new, density_new, invdeltsq, a, ap);
  density_new.FillBoundary(geom.periodicity());

  gravity->solve_Phi_data(level, geom, density_new, PhiGrav_new, a);
//  PhiGrav_new.FillBoundary(geom.periodicity());
//  const int i = 129,
//            j = 64,
//            k = 64;
//  for (amrex::MFIter mfi(KG_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
//       amrex::Array4<amrex::Real> const& arr_new = PhiGrav_new.array(mfi);
//       amrex::Array4<amrex::Real> const& arr_old = density_new.array(mfi);
//       amrex::Array4<amrex::Real> const& KG = KG_new.array(mfi);

//       printf("\n\nKG(%i, %i, %i, 0) = %e\ndensity_new(%i, %i, %i, 0) = %e\nPhiGrav_new(%i, %i, %i, 0) = %e\n\n", i, j, k, KG(i,j,k), 
//       													             i, j, k, arr_old(i, j, k), 
//       													             i, j, k, arr_new(i, j, k));
//  }
}

//amrex::Real AxNewt::advance(amrex::Real time, amrex::Real dt_old,
//                                int iteration, int ncycle) { // LSR -- what is this doing here?
//  BL_PROFILE("AxNewt::advance()");

//  amrex::Real dt = est_time_step(dt_old);

  // Print diagnostic information
//  amrex::Print() << "AxNewt::advance at time " << time << " with dt " << dt
//                 << std::endl;

//  return dt;
//}

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
  interp = &amrex::cell_bilinear_interp;

  // Establish the additional workhorse fields
  std::cout << "Adding descriptors to desc_lst..." << std::endl;

  desc_lst.addDescriptor(getState(StateType::Density_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 1, 1 /* nFields() */, interp,
                         state_data_extrap, store_in_checkpoint);

  desc_lst.addDescriptor(getState(StateType::PhiGrav_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 1 /* nextra - number of ghost cells. 0 for KGf and KGfv */, nFields(), interp,
                         state_data_extrap, store_in_checkpoint);

  // Set components
  std::cout << "Setting components..." << std::endl;
  set_scalar_bc(bc, phys_bc);

  // Establish fields
  
  desc_lst.setComponent(getState(StateType::Density_Type),
                        getField(Fields::Density), "Edens_pr", bc, bndryfunc);
  desc_lst.setComponent(getState(StateType::PhiGrav_Type),
                        getField(Fields::PhiGrav), "PhiGrav_pr", bc, bndryfunc);  // LSR -- TODO: change this to pr. Maybe PhiGrav_pr and PhiGravV_pr?
  desc_lst.setComponent(getState(StateType::PhiGrav_Type),
                        getField(Fields::PhiGravv), "PhiGravV_pr", bc, bndryfunc);

  // TODO: add derived fields with non-program units

  derive_lst.add("PhiGrav", amrex::IndexType::TheCellType(), 1, Derived::derPhiGrav, Derived::grow_box_by_one);
  derive_lst.addComponent("PhiGrav", desc_lst, getState(StateType::PhiGrav_Type), getField(Fields::PhiGrav), 1);  // LSR -- what are these two lines doing?
  derive_lst.addComponent("PhiGrav", desc_lst, getState(StateType::PhiGrav_Type), getField(Fields::PhiGravv), 1);

#ifdef TEST
  derive_lst.add("Edens_rel", amrex::IndexType::TheCellType(), 1, Derived::derEdens_rel, Derived::grow_box_by_one);
  derive_lst.addComponent("Edens_rel", desc_lst, getState(StateType::Density_Type), getField(Fields::Density), 1);  // LSR -- what are these two lines doing?

  derive_lst.add("PhiGravv", amrex::IndexType::TheCellType(), 1, Derived::derPhiGravv, Derived::grow_box_by_one);
  derive_lst.addComponent("PhiGravv", desc_lst, getState(StateType::PhiGrav_Type), getField(Fields::PhiGrav), 1);  // LSR -- what are these two lines doing?
  derive_lst.addComponent("PhiGravv", desc_lst, getState(StateType::PhiGrav_Type), getField(Fields::PhiGravv), 1);
#endif
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
  case Fields::PhiGrav:
    field = 0;
    break;
  case Fields::PhiGravv:
    field = 1;
    break;
  }
  if (field == -1) {
    std::cerr << "Invalid field requested!" << std::endl;
    amrex::Abort("Invalid field.");
  }
  return field;
}

int AxNewt::nStates() {
  // If AxKG had 1 state (KG_Type) = 0
  // we add 2 more here: Density_Type = 1, PhiGrav_Type = 2
  // => total = 3
  return 3;
}

int AxNewt::getState(StateType st) {
  int state = -1;
  switch (st) {
//  case StateType::KG_Type:
//    state = 0;
//    break;
  case StateType::Density_Type:
    state = 1;
    break;
  case StateType::PhiGrav_Type:	// LSR -- Do we want to have one state for AxNewt with three components? Keeping density and gravity separate for now but worth considering
    state = 2;
    break;
  }
  if (state == -1) {
    std::cerr << "Invalid state requested!" << std::endl;
    amrex::Abort("Invalid state.");
  }
  return state;
}

// Retrieving the general density field and override it with initial density
// field from initData() - LSR -- Why? Why not fold this into init?
MultiFab &AxNewt::get_density(bool old) {
  if (old) {
    return get_old_data(AxNewt::getState(AxNewt::StateType::Density_Type));
  } else {
    return get_new_data(AxNewt::getState(AxNewt::StateType::Density_Type));
  }
}
