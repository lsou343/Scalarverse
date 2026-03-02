#include <AxKGComov.H>
#include <KG_compute_models.H>
#include <Comoving_Full.H>

amrex::Real AxKGComov::advance (amrex::Real time,
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

    BL_PROFILE("AxKGComov::advance_KG()");
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

    amrex::MultiFab&  KG_old = get_level(level).get_old_data(getState(StateType::KG_Type));
    amrex::MultiFab&  KG_new = get_level(level).get_new_data(getState(StateType::KG_Type));

    ///// Following the Kick-Drift-Kick formulation of the Leapfrog integration algorithm:
    //
    //   v_{i+1/2} = v_i + a_i(dt/2)
    //   x_{i+1} = x_i + v_{i+1/2}dt
    //   v_{i+1} = v_{i+1/2} + a_{i+1}(dt/2)
    kick_KG(time, dt_half, KG_old, KG_new, invdeltasq);

    drift_KG(dt, KG_old, KG_new);

    // Only advance the scale-factor with the root grid
    if(level == 0)
    {
        Comoving::kick_a(dt_half, true); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
        Comoving::drift_a(dt);
        Comoving::reset_rho();
        fill_rho();
        Comoving::kick_a(dt_half, false); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
    }

    kick_KG(time+dt, dt_half, KG_new, KG_new, invdeltasq);  //N.B. The time+dt is what makes it a_{i+1} on the second go. --PH

    BL_PROFILE_VAR_STOP(KG_ADVANCE);
    
    return dt;
}


void AxKGComov::kick_KG(amrex::Real time, amrex::Real dt_half, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq) // LSR -- Not made by me, but advances field derivative
{
    static const int neighbours = 1; 
    for (amrex::FillPatchIterator 
        fpi(*this, mf_old, neighbours, time, getState(StateType::KG_Type), 0, 2);
        fpi.isValid(); ++fpi)
        {
            const amrex::Box& bx  = fpi.validbox();
            amrex::Array4<amrex::Real> const& arr_in   = fpi().array();
            amrex::Array4<amrex::Real> const& arr_old  = mf_old[fpi].array();
            amrex::Array4<amrex::Real> const& arr_new  = mf_new[fpi].array();

            amrex::ParallelFor(bx,
                               [=] AMREX_GPU_DEVICE (int i, int j, int k)
                               {

                                   amrex::Real tmp = 0.;
                                   tmp = Models::compute_acceleration(arr_in,i,j,k,getField(Fields::KGf),invdeltasq, Comoving::get_comoving_a(time), Comoving::get_comoving_ap(time), Comoving::get_comoving_app(time));
                                   // Kick 1: v_{i+1/2}    =        v_i             +   a_i*dt/2
                                   // Kick 2: v_{i+1}    =        v_{i+1/2}         +   a_{i+1}*dt/2  
                                   arr_new(i,j,k,getField(Fields::KGfv)) = arr_old(i,j,k,getField(Fields::KGfv)) + dt_half*tmp;
            });
	}
}

void AxKGComov::drift_KG(amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new)  // LSR -- Not made by me, but advances field
{
	for (amrex::MFIter mfi(mf_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
		amrex::Array4<amrex::Real> const& arr_old = mf_old.array(mfi);
		amrex::Array4<amrex::Real> const& arr_new = mf_new.array(mfi);
		const amrex::Box& bx = mfi.tilebox();
		amrex::ParallelFor(bx,
				[=] AMREX_GPU_DEVICE (int i, int j, int k)
				{
				// x_{i+1}             =   x_i                    +       v_{i+1/2} * dt
				arr_new(i,j,k,getField(Fields::KGf)) = arr_old(i,j,k,getField(Fields::KGf)) + arr_new(i,j,k,getField(Fields::KGfv))*dt;
				});
	}
}
