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

    amrex::MultiFab&  KG_old = get_level(level).get_old_data(AxKG::getState(AxKG::StateType::KG_Type));  // Wonder if this is not working properly?
    amrex::MultiFab&  KG_new = get_level(level).get_new_data(AxKG::getState(AxKG::StateType::KG_Type));
    KG_old.FillBoundary(geom.periodicity());

    amrex::MultiFab&  density_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::Density_Type));

    amrex::MultiFab&  Phi_old = get_level(level).get_old_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));  // LSR -- Do we need to call Phi_old? Or is this just slowing us down?
    amrex::MultiFab&  Phi_new = get_level(level).get_new_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type));  // Other option - never move Phi_new -> Phi_old
    MultiFab::Copy(parent->getLevel(level).get_new_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type)),
                   parent->getLevel(level).get_old_data(AxNewt::getState(AxNewt::StateType::PhiGrav_Type)),
                   0, 0, 1, 0);


    ///// Following the Kick-Drift-Kick formulation of the Leapfrog integration algorithm:
    //
    //   v_{i+1/2} = v_i + a_i(dt/2)
    //   x_{i+1} = x_i + v_{i+1/2}dt
    //   v_{i+1} = v_{i+1/2} + a_{i+1}(dt/2)

    kick_KG(time, dt_half, KG_old, KG_new, Phi_old, invdeltasq);
    KG_new.FillBoundary(geom.periodicity());

    drift_KG(dt, KG_old, KG_new);

#ifdef COMOV_FULL
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

    kick_KG(time+dt, dt_half, KG_new, KG_new, Phi_old, invdeltasq);  //N.B. The time+dt is what makes it a_{i+1} on the second go. --PH

    gravity->solve_density_data(level, KG_new, density_new, invdeltasq, a, ap);

    gravity->solve_Phi_data(level, geom, density_new, Phi_new, a);

    BL_PROFILE_VAR_STOP(KG_ADVANCE);

    return dt;
}

void AxNewt::kick_KG(amrex::Real time, amrex::Real dt_half, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, amrex::MultiFab&  Phi_old, const amrex::Real invdeltasq) // LSR -- Not made by me, but advances field derivative
{
#ifdef COMOV_FULL
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

            amrex::Array4<amrex::Real> const& arr_Phi  = Phi_old[fpi].array();

            amrex::ParallelFor(bx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k)
                               {

                                   amrex::Real tmp = 0.;
//#ifndef TEST  // Want this to be a way to decide whether or not to include gravity in EoM - TODO: implement this properly
                                   if (time > 50.) {
                                     tmp = Models::compute_acceleration(arr_in,i,j,k,AxKG::getField(AxKG::Fields::KGf),invdeltasq, a, ap, app);
//#else
                                   } else {
                                     tmp = Models::compute_acceleration(arr_in,arr_Phi,i,j,k,AxKG::getField(AxKG::Fields::KGf),invdeltasq, a, ap, app);
                                   }
//#endif
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
