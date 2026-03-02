// #include <AxKG.H>
#include <AxKGComov.H>
#include <Comoving_Full.H>
#include <KG_compute_models.H>
// #include <AMReX_ParallelDescriptor.H>

AxKGComov::~AxKGComov() {}

void AxKGComov::variable_setup() { AxKG::variable_setup(); }

void AxKGComov::init(amrex::AmrLevel &old) { AxKG::init(old); }

void AxKGComov::init() { AxKG::init(); }

void AxKGComov::initData() {
  BL_PROFILE("AxKGComov::initData()");

  // Initialize the comoving data. This must be done before calling
  // AxKG::initData since the initial conditions require H (for the velocities)
  const auto geomdata = geom.data();
  int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                 geom.Domain().length(2); // Assuming a rectangular domain

  amrex::Real V0 =
      Models::compute_model_quantity({1.}, 0., 0., 0., 0., Models::Quant::V);
  Comoving::initComov(AxKG::A, AxKG::B, AxKG::s, AxKG::r, V0, gridsize);

  // This does the bulk of the work, we only have a little more to do if we're
  // using comoving coordinates

  AxKG::initData();

  // And then this adjusts the program velocity so that the real velocity starts
  // at 0 (this is a departure from LatticeEasy, which sets the program velocity
  // to 0).
  amrex::MultiFab &KG_new = get_new_data(getState(StateType::KG_Type));

  // KG_new.plus(AxKG::r * AxKG::A * Comoving::get_comoving_ap() * AxKG::KG0,
  // getField(Fields::KGfv), 1);

  // Initialize the energy densities in the comoving space
  fill_rho();

  Comoving::set_ics(); // This resets ap and app according to the energy
                       // conservation ratio. No strictly necessary, but I think
                       // it's cleaner.
}

amrex::Real AxKGComov::est_time_step(amrex::Real dt_old) {
  BL_PROFILE("AxKG::est_time_step()");
  amrex::Real cur_time = state[State_for_Time].curTime();

  // stop simulation if final_a is defined and reached:
  if (Comoving::final_a > 0.0) {
    amrex::Real current_a = Comoving::get_comoving_a(cur_time);
    if (current_a > Comoving::final_a) {
      parent->checkPoint();
      parent->writePlotFile();
      Comoving::stop_at_final_a(Comoving::get_comoving_a(cur_time),
                                Comoving::final_a);
    }
  }
  // If fixed, just return that
  if (fixed_dt > 0)
    return fixed_dt;

  // This is just a dummy value to start with
  amrex::Real est_dt = 1.0e+200;

  if (level == 0) {
    static amrex::Real mass = AxKG::simPars[0];
    amrex::Real a = Comoving::get_comoving_a();
    const amrex::Real *dx = geom.CellSize();
    // est_dt = 0.002 * 2.0 / 3.0 * dx[0] * std::sqrt(a) / std::sqrt(3);
    est_dt = 1.0 / 2.0 * dx[0] * std::sqrt(a) / std::sqrt(3);
    if (verbose > 0)
      amrex::Print() << "Modified time step: " << est_dt << std::endl;
  }

  if (verbose && amrex::ParallelDescriptor::IOProcessor())
    std::cout << "AxKGComov::est_time_step at level " << level
              << ":  estdt = " << est_dt << '\n';

  return est_dt;
}

void AxKGComov::average_down() { AxKG::average_down(); }

void AxKGComov::post_restart() {
  if (level > 0)
    return;

  amrex::Real a = 0., ap = 0., app = 0.;
  amrex::Real a_prev = 0., ap_prev = 0., app_prev = 0.;
  amrex::Real time = 0., prev_time = 0.;
  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::string restart_file = parent->theRestartFile();
    // Format for Comoving_Full output file:
    // a
    // ap
    // app
    std::string FileName = restart_file + "/comoving_a";
    std::ifstream File;
    File.open(FileName.c_str(), std::ios::in);
    if (!File.good())
      amrex::FileOpenFailed(FileName);
    File >> a;
    File >> ap;
    File >> app;
    File >> a_prev;
    File >> ap_prev;
    File >> app_prev;
    File >> time;
    File >> prev_time;
  }

  int gridvol = geom.Domain().length(0) * geom.Domain().length(1) *
                geom.Domain().length(2); // Assuming a rectangular domain
  Comoving::restartComov(AxKG::A, AxKG::B, AxKG::s, AxKG::r, gridvol, a, ap,
                         app, a_prev, ap_prev, app_prev, time, prev_time);
  fill_rho();
}

void AxKGComov::fill_rho() {

  if (level != 0) // We only compute the energy density once, so we do it when
                  // the first (i.e., root) grid is evolving.
    return;

  amrex::MultiFab &baseMF = get_new_data(getState(StateType::KG_Type));

  // This is necessary to ensure we can compute gradients at the problem
  // boundaries
  amrex::MultiFab rhoMF(baseMF.boxArray(), baseMF.DistributionMap(), 2,
                        1); // (BoxArray, DistributionMapping, nComps, nGrow)
  rhoMF.ParallelCopy(baseMF);
  rhoMF.FillBoundary(geom.periodicity());

  const amrex::Real *dx = geom.CellSize();
  const amrex::Real invdeltasq = 1.0 / dx[0] / dx[0];
  fill_rho(rhoMF, invdeltasq);
}

// For a given MultiFab, fill the energy density.
void AxKGComov::fill_rho(amrex::MultiFab &mf, amrex::Real invdeltsq) {
  amrex::Real tmp_grad = 0., tmp_pot = 0., tmp_kin = 0.;
  for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi) {
    amrex::Array4<amrex::Real> const &arr = mf.array(mfi);
    const amrex::Box &bx = mfi.validbox();

    amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) {
      amrex::Real *tmp =
          Models::compute_rho(arr, i, j, k, AxKG::getField(AxKG::Fields::KGf),
                              invdeltsq, Comoving::get_comoving_a());
      tmp_grad += tmp[0];
      tmp_pot += tmp[1];

      amrex::Real a = Comoving::get_comoving_a(),
                  ap = Comoving::get_comoving_ap();
      amrex::Real H = ap / a;
      amrex::Real coeff = 1.;
      tmp_kin += 0.5 * coeff * arr(i, j, k, getField(Fields::KGfv)) *
                 arr(i, j, k, getField(Fields::KGfv));
      tmp_kin -= coeff * AxKG::r * arr(i, j, k, getField(Fields::KGfv)) *
                 arr(i, j, k, getField(Fields::KGf)) * H;
      tmp_kin += 0.5 * coeff * AxKG::r * AxKG::r *
                 arr(i, j, k, getField(Fields::KGf)) *
                 arr(i, j, k, getField(Fields::KGf)) * H * H;
    });
  }
  Comoving::add_to_rho(tmp_grad, tmp_pot, tmp_kin);
}
