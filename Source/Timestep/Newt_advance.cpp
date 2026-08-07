#include <AxKG.H>
#include <KG_compute_models.H>

#include <Newtonian.H>
#include <AxNewt.H>

amrex::Real AxNewt::advance (amrex::Real time,
              amrex::Real dt,
              int  iteration,
              int  ncycle)
  // Arguments:
  //    time      : the current simulation time
  //    dt        : the timestep to advance (e.g., go from time to
  //                time + dt)
  //    iteration : where we are in the current AMR subcycle.  Each
  //                level will take a number of steps to reach the
  //                final time of the coarser level below it.  This
  //                counter starts at 1
  //    ncycle    : the number of subcycles at this level
{
#ifdef INFLATION
    AxKGComov::advance(time, dt, iteration, ncycle);
    amrex::Real a = Comoving::get_comoving_a(),
                ap = Comoving::get_comoving_ap();
#else
    AxKG::advance(time, dt, iteration, ncycle);
    amrex::Real a = 1.,
                ap = 0.;
#endif

//    amrex::MultiFab&  density_old = get_old_data(AxNewt::getState(AxNewt::StateType::Density_Type));  // LSR -- I don't think I need this
    amrex::MultiFab&  KG_new = get_level(level).get_new_data(AxKG::getState(AxKG::StateType::KG_Type));

    amrex::MultiFab&  Phi_old = get_level(level).get_old_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));  // LSR -- Probably do need this though for AxKG eventually

    amrex::MultiFab&  density_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::Density_Type));
    amrex::MultiFab&  Phi_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));
    amrex::MultiFab   rhs(KG_new.boxArray(), KG_new.DistributionMap(), 1, 1);
    amrex::MultiFab   KG(KG_new.boxArray(), KG_new.DistributionMap(), 2, 1);

    KG.ParallelCopy(KG_new);
    KG.FillBoundary(geom.periodicity());

//    gravity->solve_density_data(bx, arr, fab_new, invdeltsq, a, ap);
//    gravity->solve_rhs();
//    gravity->solve_Phi_data();

    const amrex::Real *dx = geom.CellSize();
    const amrex::Real invdeltsq = 1.0 / dx[0] / dx[0];  // LSR -- what happens with mesh refinement here?

    amrex::Real phi_avg = 0.;

    for (amrex::MFIter mfi(density_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      amrex::Array4<amrex::Real> const& arr = KG.array(mfi);
      const amrex::Box &bx = mfi.tilebox();
      amrex::Array4<amrex::Real> fab_new = density_new.array(mfi);

      gravity->solve_density_data(bx, arr, fab_new, invdeltsq, a, ap);
//      printf("\n\nTest1\n\n");
    }
    rhs.ParallelCopy(density_new, 0, 0, 1, 1, 1);  // LSR -- Copy density into rhs
    rhs.FillBoundary(geom.periodicity());

    // So it seems we can't just blindly copy the answer in...
    // MultiFab::Copy(Phi_new[level], Phi_old[level], 0, 0, 1, 1);

//    Phi_new.ParallelCopy(Phi_old, 0, 0, 1, 1, 1);  // LSR -- Copy Phi_old as an initial guess for the solver - will also do Phidot eventually. This doesn't work - maybe can't change the values this way? Need to see how previous code did it

    gravity->solve_rhs(geom, rhs, Ggravity);
    gravity->solve_Phi_data(geom, rhs, Phi_new, a);

//    BL_PROFILE_VAR_STOP(NEWT_ADVANCE);

    return dt;
}
