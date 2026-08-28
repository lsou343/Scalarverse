#include <AxKG.H>
#include <AxKGComov.H>
#include <Comoving_Full.H>
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
    amrex::MultiFab::RegionTag amrlevel_tag("AmrLevel_Level_" + std::to_string(level));

    BL_PROFILE("AxNewt::advance_KG()");
    amrex::Gpu::LaunchSafeGuard lsg(true);

    // Move newData to oldData
    for (int k = 0; k < NUM_STATE; k++)
    {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    BL_PROFILE_VAR("KG_ADVANCE", KG_ADVANCE);

    if (verbose && amrex::ParallelDescriptor::IOProcessor() ){
        std::cout << "Advancing the inflaton at level " << level << " ...\n";
    }

    const amrex::Real* dx      = geom.CellSize();
    const amrex::Real invdeltasq  = 1.0 / dx[0] / dx[0];
    const amrex::Real dt_half = 0.5*dt;

    // Set up the MultiFabs

    amrex::MultiFab&  KG_old = get_level(level).get_old_data(AxKG::getState(AxKG::StateType::KG_Type));
    amrex::MultiFab&  KG_new = get_level(level).get_new_data(AxKG::getState(AxKG::StateType::KG_Type));
//    KG_old.FillBoundary(geom.periodicity()); // LSR -- TODO: only if level = 0!!! - not working?

//    for (amrex::MFIter mfi(KG_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
//       amrex::Array4<amrex::Real> const& arr_new = KG_new.array(mfi);
//       amrex::Array4<amrex::Real> const& arr_old = KG_old.array(mfi);

//       printf("\n\narr_old(129, 129, 129, 0) = %e\narr_new(129, 129, 129, 0) = %e\n\n", arr_old(129, 129, 129, 1), arr_new(129, 129, 129, 1));
//    }

    amrex::MultiFab&  density_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::Density_Type));

    amrex::MultiFab&  Phi_old = get_level(level).get_old_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));  // LSR -- Probably do need this though for AxKG eventually
    amrex::MultiFab&  Phi_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));

    ///// Following the Kick-Drift-Kick formulation of the Leapfrog integration algorithm:
    //
    //   v_{i+1/2} = v_i + a_i(dt/2)
    //   x_{i+1} = x_i + v_{i+1/2}dt
    //   v_{i+1} = v_{i+1/2} + a_{i+1}(dt/2)

    kick_KG(time, dt_half, KG_old, KG_new, Phi_old, invdeltasq);
//    KG_new.FillBoundary(geom.periodicity());  // Something like this definitely needed here since we use the ghost cells on the boundary in the next kick.
//    printf("\n\nTest1\n\n");
    drift_KG(dt, KG_old, KG_new);

#ifdef INFLATION
    // Only advance the scale-factor with the root grid
    if(level == 0)
    {
        Comoving::kick_a(dt_half, true); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
        Comoving::drift_a(dt);
        Comoving::reset_rho();
        fill_rho();
        Comoving::kick_a(dt_half, false); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
    }

    amrex::Real a = Comoving::get_comoving_a(time),
                ap = Comoving::get_comoving_ap(time);
#else
    amrex::Real a = 1.,
                ap = 0.;
#endif
//    printf("\n\nTest2\n\n");
    kick_KG(time+dt, dt_half, KG_new, KG_new, Phi_old, invdeltasq);  //N.B. The time+dt is what makes it a_{i+1} on the second go. --PH
//    KG_new.FillBoundary(geom.periodicity());

//    KG.ParallelCopy(KG_new);
//    KG.FillBoundary(geom.periodicity()); // LSR -- this probably doesn't work generally because we may have timesteps at higher levels with Dirichlet BCs. Keep for now but figure out!
//    printf("\n\nTest3\n\n");

//    for (amrex::MFIter mfi(KG_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
//       amrex::Array4<amrex::Real> const& arr_new = KG_new.array(mfi);
//       amrex::Array4<amrex::Real> const& arr_old = KG_old.array(mfi);
//       amrex::Array4<amrex::Real> const& KG_arr = KG.array(mfi);
//       printf("\n\narr_old(129, 129, 129, 0) = %e\narr_new(129, 129, 129, 0) = %e\n\n", arr_old(129, 129, 129), arr_new(129, 129, 129));
//       
//       amrex::Array4<amrex::Real> const& Phi_arr = Phi_new.array(mfi);
//       amrex::Array4<amrex::Real> const& Phi_old_arr = Phi_old.array(mfi);
//       const amrex::Box& bx = mfi.tilebox();
//       amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE (int i, int j, int k)
//         {
//             KG_arr(i,j,k,0) = arr_new(i,j,k,0);
//             KG_arr(i,j,k,1) = arr_new(i,j,k,1);
//             Phi_arr(i,j,k,0) = Phi_old_arr(i,j,k,0);
//             Phi_arr(i,j,k,1) = Phi_old_arr(i,j,k,1);
//      });
//      printf("\n\nphi_old: %e\nphi_new: %e/n/n", arr_new(64, 64, 64, 0), KG_arr(64, 64, 64, 0));
//    }
//    KG.FillBoundary(geom.periodicity()); 
//    printf("\n\nTest4\n\n");

//    printf("\n\nYour boolean variable is: %s\n\n", geom.isAllPeriodic() ? "true" : "false");
    gravity->solve_density_data(level, KG_new, density_new, invdeltasq, a, ap);
//    density_new.FillBoundary(geom.periodicity());

    // So it seems we can't just blindly copy the answer in...
//    MultiFab::Copy(Phi_new[level], Phi_old[level], 0, 0, 1, 1);
//    Phi_new.ParallelCopy(Phi_old, 0, 0, 1, 1, 1);  // LSR -- Copy Phi_old as an initial guess for the solver - will also do Phidot eventually. This doesn't work - maybe can't change the values this way? Need to see how previous code did it
//    MultiFab::Copy(parent->getLevel(level).get_new_data(
//                   AxNewt::getState(AxNewt::StateType::PhiGrav_Type)),
//                 parent->getLevel(level).get_old_data(
//                     AxNewt::getState(AxNewt::StateType::PhiGrav_Type)),
//                 0, 0, 1, 0);

    gravity->solve_Phi_data(level, geom, density_new, Phi_new, a);
//    Phi_new.FillBoundary(geom.periodicity());  // LSR -- this shouldn't be necessary if it is based on density_new, but maybe keep in case

    const int i = 129,
              j = 64,
              k = 64;
    for (amrex::MFIter mfi(KG_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
         amrex::Array4<amrex::Real> const& arr_new = Phi_new.array(mfi);
         amrex::Array4<amrex::Real> const& arr_old = density_new.array(mfi);
         amrex::Array4<amrex::Real> const& KG = KG_new.array(mfi);

//         printf("\n\nKG(%i, %i, %i, 0) = %e\nKGv(%i, %i, %i, 0) = %e\ndensity_new(%i, %i, %i, 0) = %e\nPhiGrav_new(%i, %i, %i, 0) = %e\n\n", 
//         													    i, j, k, KG(i, j, k), 
//         													    i, j, k, KG(i, j, k, 1), 
//       													            i, j, k, arr_old(i, j, k), 
//       													            i, j, k, arr_new(i, j, k));
  }

    BL_PROFILE_VAR_STOP(KG_ADVANCE);

    return dt;
}

void AxNewt::kick_KG(amrex::Real time, amrex::Real dt_half, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, amrex::MultiFab&  Phi_old, const amrex::Real invdeltasq) // LSR -- Not made by me, but advances field derivative
{
#ifdef INFLATION
    amrex::Real a = Comoving::get_comoving_a(time), 
                ap = Comoving::get_comoving_ap(time), 
                app = Comoving::get_comoving_app(time);
#else
    amrex::Real a = 1., 
                ap = 0., 
                app = 0.;
#endif

    static const int neighbours = 1; 
    for (amrex::FillPatchIterator 
        fpi(*this, mf_old, neighbours, time, AxKG::getState(AxKG::StateType::KG_Type), 0, 2);
        fpi.isValid(); ++fpi)
        {
            const amrex::Box& bx  = fpi.validbox();
            amrex::Array4<amrex::Real> const& arr_in   = fpi().array();
            amrex::Array4<amrex::Real> const& arr_old  = mf_old[fpi].array();
            amrex::Array4<amrex::Real> const& arr_new  = mf_new[fpi].array();
//            printf("\n\nTesta\n\n");
            amrex::Array4<amrex::Real> const& arr_Phi  = Phi_old[fpi].array();  // LSR -- How do we get access to Phi data in this region? Works for density so what is different?
//            printf("\n\nTestb\n\n");

            amrex::ParallelFor(bx,
                               [&] AMREX_GPU_DEVICE (int i, int j, int k)
                               {

                                   amrex::Real tmp = 0.;
#ifndef TEST  // Want this to be a way to decide whether or not to include gravity in EoM - TODO: implement this properly
                                   tmp = Models::compute_acceleration(arr_old,i,j,k,AxKG::getField(AxKG::Fields::KGf),invdeltasq, a, ap, app);
#else
                                   tmp = Models::compute_acceleration(arr_old,arr_Phi,i,j,k,AxKG::getField(AxKG::Fields::KGf),invdeltasq, a, ap, app);
#endif
                                   // Kick 1: v_{i+1/2}    =        v_i             +   a_i*dt/2
                                   // Kick 2: v_{i+1}    =        v_{i+1/2}         +   a_{i+1}*dt/2  
                                   arr_new(i,j,k,AxKG::getField(AxKG::Fields::KGfv)) = arr_old(i,j,k,AxKG::getField(AxKG::Fields::KGfv)) + dt_half*tmp;
            });
//            printf("\n\nphi: %e\nphidot: %e\nphigrav: %e\n\n", 
//            arr_old(64,64,64,0), 
//            arr_new(64,64,64,1), 
//            arr_Phi(64,64,64,0));
	}
}

void AxNewt::drift_KG(amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new)  // LSR -- Not made by me, but advances field
{
	for (amrex::MFIter mfi(mf_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
		amrex::Array4<amrex::Real> const& arr_old = mf_old.array(mfi);
		amrex::Array4<amrex::Real> const& arr_new = mf_new.array(mfi);
		const amrex::Box& bx = mfi.tilebox();
		amrex::ParallelFor(bx,
				[&] AMREX_GPU_DEVICE (int i, int j, int k)
				{
				// x_{i+1}             =   x_i                    +       v_{i+1/2} * dt
				arr_new(i,j,k,AxKG::getField(AxKG::Fields::KGf)) = arr_old(i,j,k,AxKG::getField(AxKG::Fields::KGf)) + arr_new(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*dt;
				});
	}
}
