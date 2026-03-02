#include <AxSCH.H>
#include <SCHDerive.H>
#include <bc_fill.H>

// Add relevant headers for SCH-specific components
using namespace amrex;
namespace {
int scalar_bc[] = {
    // Define the boundary conditions specifically for SCH
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
} // namespace

int BaseAx::NUM_STATE = AxSCH::nStates();

AxSCH::AxSCH() {
  BL_PROFILE("AxSCH::AxSCH()");
  fine_mask = 0;
}

AxSCH::AxSCH(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
             const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
             amrex::Real time)
    : BaseAx(papa, lev, level_geom, bl, dm, time) {
  BL_PROFILE("AxSCH::AxSCH(Amr)");
}

void AxSCH::init(AmrLevel &old) {
  BaseAx::init(old);

  amrex::MultiFab &SCH_new = get_new_data(getState(StateType::SCH_Type));

  AxSCH *old_level = static_cast<AxSCH *>(&old);
  amrex::Real cur_time = old_level->state[State_for_Time].curTime();

  FillPatch(old, SCH_new, 0, cur_time, getState(StateType::SCH_Type), 0,
            nFields());

  amrex::Gpu::Device::streamSynchronize();
}
//
// This version inits the data on a new level that did not
// exist before regridding.
//
void AxSCH::init() {
  BaseAx::init();

  amrex::Real cur_time = static_cast<AxSCH *>(&get_level(level - 1))
                             ->state[State_for_Time]
                             .curTime();

  amrex::MultiFab &SCH_new = get_new_data(getState(StateType::SCH_Type));
  FillCoarsePatch(SCH_new, 0, cur_time, getState(StateType::SCH_Type), 0,
                  SCH_new.nComp());

  // We set dt to be large for this new level to avoid screwing up
  // computeNewDt.
  parent->setDtLevel(1.e100, level);
}

void AxSCH::initData() {
  BL_PROFILE("AxSCH::initData()");

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

amrex::Real AxSCH::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxSCH::est_time_step()");

  // In case we have a simple fixed time step.
  if (BaseAx::fixed_dt > 0)
    return BaseAx::fixed_dt;

  amrex::Real est_dt = 1.0e+200;

  if (vonNeumann_dt > 0) {

    amrex::Real cur_time = state[State_for_Time].curTime();
    // Real a = get_comoving_a(cur_time);
    amrex::Real a = 1;
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
  }
  return est_dt;
}

void AxSCH::average_down() {
  // This is only implemented in the derived class because we have to specify
  // the state type. However, for hydro code there is more to be done, and an
  // overwrite of average_down(state_type) is in order.  --PH
  amrex::Print() << "AxSCH::average_down() level:" << level << std::endl;

  BL_PROFILE("AxSCH::average_down()");
  if (level == parent->finestLevel())
    return;

  BaseAx::average_down(getState(StateType::SCH_Type));
}

void AxSCH::variable_setup() {
  BaseAx::variable_setup();

  // Get options, set phys_bc  --- This is necessary because these are all
  // static functions, so there's no actual inheritance!
  AxSCH::read_params();

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

  amrex::Interpolater *SCHinterp;
  SCHinterp = &amrex::cell_bilinear_interp;

  desc_lst.addDescriptor(getState(StateType::SCH_Type),
                         amrex::IndexType::TheCellType(),
                         amrex::StateDescriptor::Point, 0, nFields(), SCHinterp,
                         state_data_extrap, store_in_checkpoint);

  set_scalar_bc(bc, phys_bc);

  // Track the field for time-stepping
  desc_lst.setComponent(getState(StateType::SCH_Type), 0, "Dens", bc,
                        bndryfunc);
  desc_lst.setComponent(getState(StateType::SCH_Type), 1, "SCHf_Re", bc,
                        bndryfunc);
  desc_lst.setComponent(getState(StateType::SCH_Type), 2, "SCHf_Im", bc,
                        bndryfunc);
  desc_lst.setComponent(getState(StateType::SCH_Type), 3, "Phase", bc,
                        bndryfunc);

  // Establish the derived fields
  derive_lst.add("SCHf_Re_derived", amrex::IndexType::TheCellType(), 1,
                 Derived::derSCHf_Re, Derived::grow_box_by_one);
  derive_lst.addComponent("SCHf_Re_derived", desc_lst,
                          getState(StateType::SCH_Type),
                          getField(Fields::SCHf_Re), 1);
}

int AxSCH::nFields() {
  return 4; // Density, SCHf_Re, SCHf_Im, Phase
}

int AxSCH::getField(Fields f) {
  switch (f) {
  case Fields::Dens:
    return 0;
  case Fields::SCHf_Re:
    return 1;
  case Fields::SCHf_Im:
    return 2;
  case Fields::Phase:
    return 3;
  }
  return -1;
}

AxSCH::Fields AxSCH::getField(int f) {
  switch (f) {
  case 0:
    return Fields::Dens;
  case 1:
    return Fields::SCHf_Re;
  case 2:
    return Fields::SCHf_Im;
  case 3:
    return Fields::Phase;
  }
  return Fields::Dens; // TODO: This should be an error value.
}

int AxSCH::nStates() {
  return 1; // SCh_Type
}

int AxSCH::getState(StateType st) {
  switch (st) {
  // NOTE: This must be the state used to track the time variable.
  case StateType::SCH_Type:
    return 0;
  }
  return -1;
}

AxSCH::StateType AxSCH::getState(int st) {
  switch (st) {
  // NOTE: This must be the state used to track the time variable.
  case 0:
    return StateType::SCH_Type;
  }
  return StateType::SCH_Type; // TODO: This should be an error value.
}

int AxSCH::getIC(ICType it) {
  switch (it) {
  case ICType::test:
    return 0;
  case ICType::plain_wave:
    return 1;
  case ICType::gaussian:
    return 2;
  case ICType::KGfield:
    return 3;
  case ICType::linearperurbation:
    return 4;
  }
  return -1;
}

AxSCH::ICType AxSCH::getIC(int it) {
  switch (it) {
  case 0:
    return ICType::test;
  case 1:
    return ICType::plain_wave;
  case 2:
    return ICType::gaussian;
  case 3:
    return ICType::KGfield;
  case 4:
    return ICType::linearperurbation;
  }

  return ICType::test; // TODO: This should be an error value.
}

void AxSCH::prob_param_fill(
    amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_params) {
  prob_params[0] = getIC(ic);

  switch (ic) {
  case ICType::test:
    prob_params[1] = SCH0;
    break;

  case ICType::plain_wave:
    prob_params[1] = SCH0;
    prob_params[2] = SCH_k0;
    prob_params[3] = Phase0;
    break;
  case ICType::gaussian:
    prob_params[1] = SCH0;
    prob_params[2] = sigma;
    prob_params[3] = SCH_k0;
    break;
  case ICType::KGfield:
    prob_params[1] = KGm;
    prob_params[2] = KGA;
    prob_params[3] = KGB;
    prob_params[4] = KGr;
    prob_params[5] = KGs;
    break;
  case ICType::linearperurbation:
    prob_params[1] = SCH0;
    prob_params[2] = SCH_k0;
    prob_params[3] = Phase0;
    break;
  }
}

#ifdef BL_USE_MPI

void AxSCH::prob_initdata_mom_on_mf(
    amrex::MultiFab &mf, amrex::Geometry const &geom,
    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {
  // We have real fields in position space, so we need to enforce f_k = f_{-k},
  // which cannot be done in BaseAx (nor should it be), so we need to
  // re-implement this here.

  // BL_PROFILE("AxSCH::prob_initdata_mom_on_mf")
  //
  // const auto geomdata = geom.data();
  // amrex::MultiFab fillK(mf.boxArray(), mf.DistributionMap(), mf.nComp()*2,
  // 0);
  //
  // // Initialize the Fourier data
  // for (amrex::MFIter mfi(fillK,amrex::TilingIfNotGPU()); mfi.isValid();
  // ++mfi)
  // {
  //     const amrex::Box& bx  = mfi.tilebox();
  //     const auto fab_new = fillK.array(mfi);
  //     amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k)
  //     noexcept
  //     // NOTE THE & instead of =!! This takes references for external
  //     variables as opposed to making local copies.
  //     {
  //         prob_initdata_mom(i, j ,k, fab_new, geomdata, prob_param);
  //
  //         // Set the 0-mode to 0 since we add it in later
  //         if(i == 0 && j == 0 && k == 0)
  //         {
  //             for(int comp = 0; comp < fillK.nComp(); comp++)
  //                 fab_new(0, 0, 0, comp) = 0.;  // Set the 0-mode to 0 since
  //                 we add it in later
  //         }
  //     });
  // }
  //
  // // Arrange the Fourier data to ensure a purely real transform
  // for(int i = 0; i < mf.nComp(); i++)
  //     DFFTUtils::makeReal(fillK, 2*i);
  //
  // // And perform the DFFT.
  // amrex::MultiFab back;
  // for(int comp = 0; comp < mf.nComp(); comp++)
  // {
  //     back = DFFTUtils::backward_dfft(fillK, 2*comp, true);
  //     back.mult(1./back.boxArray().numPts());
  //
  //     // We're just keeping the real part of the DFT---the imaginary part
  //     should almost vanish. mf.ParallelCopy(back, 0, comp, 1, back.nGrow(),
  //     mf.nGrow(), geom.periodicity(), amrex::FabArrayBase::COPY);
  // }
  //
  // mf.FillBoundary(geom.periodicity());
}
#endif // BL_USE_MPI

void AxSCH::errorEst(amrex::TagBoxArray &tags, int clearval, int tagval,
                     amrex::Real time, int n_error_buf, int ngrow) {
  BL_PROFILE("AxSCH::errorEst()");

  for (int j = 0; j < errtags.size(); ++j) {
    std::unique_ptr<amrex::MultiFab> mf;
    if (errtags[0].Field() != std::string()) {

      // Can't seem to make relative energy density into a derived field (needs
      // the full MultiFab, but derived field calculations only get FABs)
      if (errtags[0].Field() == "EdensRel") {
        mf = std::unique_ptr<amrex::MultiFab>(
            derive("Edens", time, errtags[j].NGrow()));

        int gridsize = get_level(0).Geom().Domain().length(0) *
                       geom.Domain().length(1) * geom.Domain().length(2);
        amrex::Real avE =
            static_cast<AxSCH *>(&get_level(0))
                ->derive("Edens", time, errtags[j].NGrow())
                ->sum() /
            gridsize; // Need to use level 0 to calculate the average
        mf->mult(1. / avE);
      } else {
        mf = std::unique_ptr<amrex::MultiFab>(
            derive(errtags[j].Field(), time, errtags[j].NGrow()));
      }
    }
    errtags[j](tags, mf.get(), clearval, tagval, time, level, geom);
  }
}

// function to read the KG MultiFab from the plotfile directory
void AxSCH::read_KG_MultiFab(std::string mfDirName, int nghost, MultiFab &mf) {
  if (level > 0 && nghost > 0) {
    std::cout << "Are sure you want to do what you are doing?" << std::endl;
    amrex::Abort();
  }

  MultiFab mf_read;
  if (!mfDirName.empty() && mfDirName[mfDirName.length() - 1] != '/')
    mfDirName += '/';
  std::string Level = amrex::Concatenate("Level_", level, 1);
  mfDirName.append(Level);
  mfDirName.append("/Cell");

  VisMF::Read(mf_read, mfDirName.c_str());

  if (ParallelDescriptor::IOProcessor())
    std::cout << "KG multifab read" << '\n';

  if (mf_read.contains_nan()) {
    for (int i = 0; i < mf_read.nComp(); i++) {
      if (mf_read.contains_nan(i, 1)) {
        std::cout << "Found NaNs in read_mf in component " << i << ". "
                  << std::endl;
        amrex::Abort("AxSCH::read_KG_MultiFab: Your initial read multifab "
                     "contain NaNs!");
      }
    }
  }
  const auto &ba = parent->boxArray(level);
  const auto &dm = parent->DistributionMap(level);
  const auto &ba_read = mf_read.boxArray();
  int nc = mf_read.nComp();
  // if we don't use a cic scheme for the initial conditions,
  // we can safely set the number of ghost cells to 0
  // for multilevel ICs we can't use ghostcells
  mf.define(ba, dm, nc, nghost);
  mf.MultiFab::ParallelCopy(mf_read, 0, 0, nc, 0, 0);

  if (!((ba.contains(ba_read) && ba_read.contains(ba)))) {
    if (ParallelDescriptor::IOProcessor()) {
      std::cout << "ba      :" << ba << std::endl;
      std::cout << "ba_read :" << ba_read << std::endl;
      std::cout
          << "Read mf and simulation setup mf DO NOT cover the same domain!"
          << std::endl;
    }
    ParallelDescriptor::Barrier();
    if (ParallelDescriptor::IOProcessor()) {
      amrex::Abort();
    }
  }
  mf_read.clear();

  mf.FillBoundary();
  mf.EnforcePeriodicity(geom.periodicity());

  if (mf.contains_nan()) {
    for (int i = 0; i < mf.nComp(); i++) {
      if (mf.contains_nan(i, 1, nghost)) {
        std::cout << "Found NaNs in component " << i << ". " << std::endl;
        amrex::Abort(
            "AxSCH::read_KG_MultiFab: Your initial multifab contain NaNs!");
      }
    }
  }
}
