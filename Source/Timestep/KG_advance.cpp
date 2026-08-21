#include <AxKG.H>
#include <KG_compute_models.H>

amrex::Real AxKG::advance (amrex::Real time,
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

    BL_PROFILE("AxKG::advance_KG()");
    amrex::Gpu::LaunchSafeGuard lsg(true);

    for (int k = 0; k < NUM_STATE; k++)
    {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    BL_PROFILE_VAR("KG_ADVANCE", KG_ADVANCE);

    if (verbose && amrex::ParallelDescriptor::IOProcessor() ){
    std::cout << "Advancing the inflaton at level " << level <<  "...\n";
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
    int gridsize = geom.Domain().length(0)*geom.Domain().length(1)*geom.Domain().length(2); // Assuming a rectangular domain

    amrex::Real avPhi = KG_old.sum()/gridsize;
    // amrex::Real avPhi = KG_new.sum()/gridsize;
    // std::cout << amrex::ParallelDescriptor::MyProc() << ": " << "First avPhi  = " << avPhi << std::endl;
    kick_KG(time, dt_half, KG_old, KG_new, invdeltasq);
    avPhi = KG_new.sum()/gridsize;
    // std::cout << amrex::ParallelDescriptor::MyProc() << ": " << "Second avPhi = " << avPhi << std::endl;
    drift_KG(dt, KG_old, KG_new);
    avPhi = KG_new.sum()/gridsize;
    // std::cout << amrex::ParallelDescriptor::MyProc() << ": " << "Third avPhi  = " << avPhi << std::endl;
    kick_KG(time+dt, dt_half, KG_new, KG_new, invdeltasq);  //N.B. The time+dt is what makes it a_{i+1} on the second go. --PH
    avPhi = KG_new.sum()/gridsize;
    // std::cout << amrex::ParallelDescriptor::MyProc() << ": " << "Fourth avPhi = " << avPhi << std::endl;
    /* } */
    ///////////

    BL_PROFILE_VAR_STOP(KG_ADVANCE);

    return dt;
}

void AxKG::kick_KG(amrex::Real time, amrex::Real dt_half, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq)
{
    static const int neighbours = 1; // This should probably be a parameter later.
	////// TODO: Figure out what this block does line-by-line --PH ///////////////
	for (amrex::FillPatchIterator 
			fpi(*this, mf_old, neighbours, time, getState(StateType::KG_Type), 0, 2);
			fpi.isValid(); ++fpi)
	{
		const amrex::Box& bx  = fpi.validbox();
		amrex::Array4<amrex::Real> const& arr_in   = fpi().array();
		amrex::Array4<amrex::Real> const& arr_old  = mf_old[fpi].array();
		amrex::Array4<amrex::Real> const& arr_new  = mf_new[fpi].array();
	//////////////////////////
		/* Real lprs = 1.0; */
		/* if(Nyx::prsstring) */
		/* 	lprs = pow(Nyx::msa/time,2); */
		/* else{ */
		/* 	if(Nyx::string_stop_time<=0.0) amrex::Abort("prsstring needs string_stop_time>0"); */
		/* 	lprs = pow(Nyx::msa/Nyx::string_stop_time,2); */
		/* } */

        amrex::ParallelFor(bx,
				[=] AMREX_GPU_DEVICE (int i, int j, int k)
				{
                amrex::Real tmp = Models::compute_acceleration(arr_in,i,j,k,getField(Fields::KGf),invdeltasq, 0, 0, 0);
				// Kick 1: v_{i+1/2}    =        v_i             +   a_i*dt/2
				// Kick 2: v_{i+1}    =        v_{i+1/2}         +   a_{i+1}*dt/2  
				arr_new(i,j,k,getField(Fields::KGfv)) = arr_old(i,j,k,getField(Fields::KGfv)) + dt_half
					// *(compute_acceleration(arr_in,i,j,k,Fields::KGf,invdeltasq, a, ap));
					*tmp;
				// arr_new(i,j,k,getField(Fields::KGfv)) = arr_old(i,j,k,getField(Fields::KGfv)); // Don't evolve
				});
	}
}

void AxKG::drift_KG(amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new)
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
				// arr_new(i,j,k,getField(Fields::KGf)) = arr_old(i,j,k,getField(Fields::KGf));  // Don't evolve
				});
	}
}
