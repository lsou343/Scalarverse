#include <AxSCH.H>
//#include <Comoving_Full.H>
#include <AMReX_MultiFabUtil.H>

amrex::Real AxSCH::advance (amrex::Real time,
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

    // Creates a tag be able to track how long it takes
    amrex::MultiFab::RegionTag amrlevel_tag("AmrLevel_Level_" + std::to_string(level));

    // tracks performance of the function
    BL_PROFILE("AxSCH::advance_SCH()");
    amrex::Gpu::LaunchSafeGuard lsg(true);

    // Move newData to oldData to be able to overwrite newData
    for (int k = 0; k < NUM_STATE; k++)
    {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    BL_PROFILE_VAR("SCH_ADVANCE", SCH_ADVANCE);

    if (verbose && amrex::ParallelDescriptor::IOProcessor() ){
        std::cout << "Advancing the Sch field at level " << level << " ...\n";
    }
    
	//void initComov(amrex::Real A, amrex::Real B, amrex::Real s, amrex::Real r, amrex::Real V0, int gridvol);
	//Comoving::initComov(1.0, 1.0, 1.0, 1.0, 1.0, 1);
	// calcualte scale factors
/*     const amrex::Real a_old = Comoving::get_comoving_a(time);
    const amrex::Real a_half = Comoving::get_comoving_a(time+0.5*dt);
    const amrex::Real a_new = Comoving::get_comoving_a(time+dt);
 */
    const amrex::Real a_old = 1;
    const amrex::Real a_half = 1;
    const amrex::Real a_new = 1;

	// Print get_comiving_a(time) to check if the function is working
/* 	amrex::Print() << "a_old: " << Comoving::get_comoving_a(time) << '\n';
	amrex::Print() << "a_new: " << Comoving::get_comoving_a(time+dt) << '\n'; */

    // define cell size dx
    const amrex::Real* dx      = geom.CellSize();


    // calculate 1/a^2dx for different times
    const amrex::Real invdeltasq_old  = 1.0 / ( a_old  * dx[0] ) / ( a_old  * dx[0] );                                                                                                       
    const amrex::Real invdeltasq_half = 1.0 / ( a_half * dx[0] ) / ( a_half * dx[0] );                                                                                                       
    const amrex::Real invdeltasq_new  = 1.0 / ( a_new  * dx[0] ) / ( a_new  * dx[0] );


	const amrex::Real dt_half = 0.5*dt;

    amrex::MultiFab&  SCH_old = get_level(level).get_old_data(getState(StateType::SCH_Type));
    // print SCH_old
    // print example amrex::Print() << "Initializing the data at level " << level << '\n';
   // amrex::Print() << "SCH_old: " << SCH_old.boxArray() << '\n';
    //amrex::IntVect cell{0, 0, 0};
   
    //print_state(SCH_old,cell);

    amrex::MultiFab&  SCH_new = get_level(level).get_new_data(getState(StateType::SCH_Type));   
    //print_state(SCH_new,cell);

#ifdef DEBUG
    if (SCH_old.contains_nan(0, SCH_old.nComp(), 0))
      {
        for (int i = 0; i < SCH_old.nComp(); i++)
          {
            if (SCH_old.contains_nan(i,1,0))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("SCH_old has NaNs in this component::AxSCH advance()");
              }
          }
      }
#endif
    ///// Following the Runge-Kutta formulation for integration:
    //
    //   k1 = f(x_n, t_n)
    //   k2 = f(x_n + 0.5*k1*dt, t_n + 0.5*dt)
    //   k3 = f(x_n + 0.5*k2*dt, t_n + 0.5*dt)
    //   k4 = f(x_n + k3*dt, t_n + dt)
    //   x_{n+1} = x_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

#ifdef GRAVITY
    amrex::MultiFab& Phi_old = get_old_data(PhiGrav_Type);
    for (amrex::FillPatchIterator
	   fpi(*this, SCH_old, 4, time, getState(StateType::SCH_Type),   0, AxSCH::nFields()),
	   pfpi(*this, Phi_old, 3, time, PhiGrav_Type, 0, 1);
	 fpi.isValid() && pfpi.isValid(); ++fpi,++pfpi)
#else
    for (amrex::FillPatchIterator
	   fpi(*this, SCH_old, 4, time, getState(StateType::SCH_Type),   0, AxSCH::nFields());
	 fpi.isValid(); ++fpi)
#endif
      {
	const amrex::Box& bxfour  = fpi.validbox();
	const amrex::Box& bxthree = grow(bxfour,1);
	const amrex::Box& bxtwo   = grow(bxthree,1);
	const amrex::Box& bxone   = grow(bxtwo,1);
	auto const sch_in   = fpi().array();
	auto const sch_out  = SCH_new[fpi].array();

#ifdef GRAVITY
	auto const phi     = pfpi().array();
#endif
	amrex::FArrayBox kr_one_fab(bxone);
	amrex::Array4<amrex::Real> const kr_one = kr_one_fab.array();
	amrex::FArrayBox ki_one_fab(bxone);
	amrex::Array4<amrex::Real> const ki_one = ki_one_fab.array();
	//how to call AxSCH::getField(Fields::SCHf_Im)_im outside of the following loop to make it a variable?
	const int field_schf_im = AxSCH::getField(Fields::SCHf_Im);
	const int field_schf_re = AxSCH::getField(Fields::SCHf_Re);

	ParallelFor(bxone,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     kr_one(i,j,k) =-hbaroverm*(6.0*sch_in(i+1,j,k,field_schf_im)
							+6.0*sch_in(i-1,j,k,field_schf_im)
							+6.0*sch_in(i,j+1,k,field_schf_im)
							+6.0*sch_in(i,j-1,k,field_schf_im)
							+6.0*sch_in(i,j,k+1,field_schf_im)
							+6.0*sch_in(i,j,k-1,field_schf_im)
							+3.0*sch_in(i+1,j+1,k,field_schf_im)
							+3.0*sch_in(i+1,j-1,k,field_schf_im)
							+3.0*sch_in(i-1,j+1,k,field_schf_im)
							+3.0*sch_in(i-1,j-1,k,field_schf_im)
							+3.0*sch_in(i+1,j,k+1,field_schf_im)
							+3.0*sch_in(i+1,j,k-1,field_schf_im)
							+3.0*sch_in(i-1,j,k+1,field_schf_im)
							+3.0*sch_in(i-1,j,k-1,field_schf_im)
							+3.0*sch_in(i,j+1,k+1,field_schf_im)
							+3.0*sch_in(i,j+1,k-1,field_schf_im)
							+3.0*sch_in(i,j-1,k+1,field_schf_im)
							+3.0*sch_in(i,j-1,k-1,field_schf_im)
							+2.0*sch_in(i+1,j+1,k+1,field_schf_im)
							+2.0*sch_in(i+1,j+1,k-1,field_schf_im)
							+2.0*sch_in(i+1,j-1,k+1,field_schf_im)
							+2.0*sch_in(i+1,j-1,k-1,field_schf_im)
							+2.0*sch_in(i-1,j+1,k+1,field_schf_im)
							+2.0*sch_in(i-1,j+1,k-1,field_schf_im)
							+2.0*sch_in(i-1,j-1,k+1,field_schf_im)
							+2.0*sch_in(i-1,j-1,k-1,field_schf_im)
							-88.0*sch_in(i,j,k,field_schf_im))
			       *invdeltasq_old/52.0
#ifdef GRAVITY
			       - phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_im))
#endif
			       ;

			     ki_one(i,j,k) = hbaroverm*(6.0*sch_in(i+1,j,k,field_schf_re)
							+6.0*sch_in(i-1,j,k,field_schf_re)
							+6.0*sch_in(i,j+1,k,field_schf_re)
							+6.0*sch_in(i,j-1,k,field_schf_re)
							+6.0*sch_in(i,j,k+1,field_schf_re)
							+6.0*sch_in(i,j,k-1,field_schf_re)
							+3.0*sch_in(i+1,j+1,k,field_schf_re)
							+3.0*sch_in(i+1,j-1,k,field_schf_re)
							+3.0*sch_in(i-1,j+1,k,field_schf_re)
							+3.0*sch_in(i-1,j-1,k,field_schf_re)
							+3.0*sch_in(i+1,j,k+1,field_schf_re)
							+3.0*sch_in(i+1,j,k-1,field_schf_re)
							+3.0*sch_in(i-1,j,k+1,field_schf_re)
							+3.0*sch_in(i-1,j,k-1,field_schf_re)
							+3.0*sch_in(i,j+1,k+1,field_schf_re)
							+3.0*sch_in(i,j+1,k-1,field_schf_re)
							+3.0*sch_in(i,j-1,k+1,field_schf_re)
							+3.0*sch_in(i,j-1,k-1,field_schf_re)
							+2.0*sch_in(i+1,j+1,k+1,field_schf_re)
							+2.0*sch_in(i+1,j+1,k-1,field_schf_re)
							+2.0*sch_in(i+1,j-1,k+1,field_schf_re)
							+2.0*sch_in(i+1,j-1,k-1,field_schf_re)
							+2.0*sch_in(i-1,j+1,k+1,field_schf_re)
							+2.0*sch_in(i-1,j+1,k-1,field_schf_re)
							+2.0*sch_in(i-1,j-1,k+1,field_schf_re)
							+2.0*sch_in(i-1,j-1,k-1,field_schf_re)
							-88.0*sch_in(i,j,k,field_schf_re))
			       *invdeltasq_old/52.0
#ifdef GRAVITY
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re))
#endif
			       ;
			       
			      });

	amrex::FArrayBox kr_two_fab(bxtwo);
	amrex::Array4<amrex::Real> const kr_two = kr_two_fab.array();
	amrex::FArrayBox ki_two_fab(bxtwo);
	amrex::Array4<amrex::Real> const ki_two = ki_two_fab.array();

	ParallelFor(bxtwo,
 			   [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     kr_two(i,j,k) =-hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_im)+ki_one(i+1,j,k)*dt/2.0)
							+6.0*(sch_in(i-1,j,k,field_schf_im)+ki_one(i-1,j,k)*dt/2.0)
							+6.0*(sch_in(i,j+1,k,field_schf_im)+ki_one(i,j+1,k)*dt/2.0)
							+6.0*(sch_in(i,j-1,k,field_schf_im)+ki_one(i,j-1,k)*dt/2.0)
							+6.0*(sch_in(i,j,k+1,field_schf_im)+ki_one(i,j,k+1)*dt/2.0)
							+6.0*(sch_in(i,j,k-1,field_schf_im)+ki_one(i,j,k-1)*dt/2.0)
							+3.0*(sch_in(i+1,j+1,k,field_schf_im)+ki_one(i+1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j-1,k,field_schf_im)+ki_one(i+1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j+1,k,field_schf_im)+ki_one(i-1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j-1,k,field_schf_im)+ki_one(i-1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j,k+1,field_schf_im)+ki_one(i+1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i+1,j,k-1,field_schf_im)+ki_one(i+1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k+1,field_schf_im)+ki_one(i-1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k-1,field_schf_im)+ki_one(i-1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k+1,field_schf_im)+ki_one(i,j+1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k-1,field_schf_im)+ki_one(i,j+1,k-1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k+1,field_schf_im)+ki_one(i,j-1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k-1,field_schf_im)+ki_one(i,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_im)+ki_one(i+1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_im)+ki_one(i+1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_im)+ki_one(i+1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_im)+ki_one(i+1,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_im)+ki_one(i-1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_im)+ki_one(i-1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_im)+ki_one(i-1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_im)+ki_one(i-1,j-1,k-1)*dt/2.0)
							-88.0*(sch_in(i,j,k,field_schf_im)+ki_one(i,j,k)*dt/2.0))
			       *invdeltasq_half/52.0
#ifdef GRAVITY
			       - phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_im)+ki_one(i,j,k)*dt/2.0)
#endif
			       ;

			     ki_two(i,j,k) = hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_re)+kr_one(i+1,j,k)*dt/2.0)
							+6.0*(sch_in(i-1,j,k,field_schf_re)+kr_one(i-1,j,k)*dt/2.0)
							+6.0*(sch_in(i,j+1,k,field_schf_re)+kr_one(i,j+1,k)*dt/2.0)
							+6.0*(sch_in(i,j-1,k,field_schf_re)+kr_one(i,j-1,k)*dt/2.0)
							+6.0*(sch_in(i,j,k+1,field_schf_re)+kr_one(i,j,k+1)*dt/2.0)
							+6.0*(sch_in(i,j,k-1,field_schf_re)+kr_one(i,j,k-1)*dt/2.0)
							+3.0*(sch_in(i+1,j+1,k,field_schf_re)+kr_one(i+1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j-1,k,field_schf_re)+kr_one(i+1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j+1,k,field_schf_re)+kr_one(i-1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j-1,k,field_schf_re)+kr_one(i-1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j,k+1,field_schf_re)+kr_one(i+1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i+1,j,k-1,field_schf_re)+kr_one(i+1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k+1,field_schf_re)+kr_one(i-1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k-1,field_schf_re)+kr_one(i-1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k+1,field_schf_re)+kr_one(i,j+1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k-1,field_schf_re)+kr_one(i,j+1,k-1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k+1,field_schf_re)+kr_one(i,j-1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k-1,field_schf_re)+kr_one(i,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_re)+kr_one(i+1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_re)+kr_one(i+1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_re)+kr_one(i+1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_re)+kr_one(i+1,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_re)+kr_one(i-1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_re)+kr_one(i-1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_re)+kr_one(i-1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_re)+kr_one(i-1,j-1,k-1)*dt/2.0)
							-88.0*(sch_in(i,j,k,field_schf_re)+kr_one(i,j,k)*dt/2.0))
			       *invdeltasq_half/52.0
#ifdef GRAVITY
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_one(i,j,k)*dt/2.0)
#endif
			       ;

			       });

	amrex::FArrayBox kr_three_fab(bxthree);
	amrex::Array4<amrex::Real> const kr_three = kr_three_fab.array();
	amrex::FArrayBox ki_three_fab(bxthree);
	amrex::Array4<amrex::Real> const ki_three = ki_three_fab.array();

	ParallelFor(bxthree,
 			   [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     kr_three(i,j,k) =-hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_im)+ki_two(i+1,j,k)*dt/2.0)
							+6.0*(sch_in(i-1,j,k,field_schf_im)+ki_two(i-1,j,k)*dt/2.0)
							+6.0*(sch_in(i,j+1,k,field_schf_im)+ki_two(i,j+1,k)*dt/2.0)
							+6.0*(sch_in(i,j-1,k,field_schf_im)+ki_two(i,j-1,k)*dt/2.0)
							+6.0*(sch_in(i,j,k+1,field_schf_im)+ki_two(i,j,k+1)*dt/2.0)
							+6.0*(sch_in(i,j,k-1,field_schf_im)+ki_two(i,j,k-1)*dt/2.0)
							+3.0*(sch_in(i+1,j+1,k,field_schf_im)+ki_two(i+1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j-1,k,field_schf_im)+ki_two(i+1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j+1,k,field_schf_im)+ki_two(i-1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j-1,k,field_schf_im)+ki_two(i-1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j,k+1,field_schf_im)+ki_two(i+1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i+1,j,k-1,field_schf_im)+ki_two(i+1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k+1,field_schf_im)+ki_two(i-1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k-1,field_schf_im)+ki_two(i-1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k+1,field_schf_im)+ki_two(i,j+1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k-1,field_schf_im)+ki_two(i,j+1,k-1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k+1,field_schf_im)+ki_two(i,j-1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k-1,field_schf_im)+ki_two(i,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_im)+ki_two(i+1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_im)+ki_two(i+1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_im)+ki_two(i+1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_im)+ki_two(i+1,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_im)+ki_two(i-1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_im)+ki_two(i-1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_im)+ki_two(i-1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_im)+ki_two(i-1,j-1,k-1)*dt/2.0)
							-88.0*(sch_in(i,j,k,field_schf_im)+ki_two(i,j,k)*dt/2.0))
			       *invdeltasq_half/52.0
#ifdef GRAVITY
			       - phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_im)+ki_two(i,j,k)*dt/2.0)
#endif
			       ;

			     ki_three(i,j,k) = hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_re)+kr_two(i+1,j,k)*dt/2.0)
							+6.0*(sch_in(i-1,j,k,field_schf_re)+kr_two(i-1,j,k)*dt/2.0)
							+6.0*(sch_in(i,j+1,k,field_schf_re)+kr_two(i,j+1,k)*dt/2.0)
							+6.0*(sch_in(i,j-1,k,field_schf_re)+kr_two(i,j-1,k)*dt/2.0)
							+6.0*(sch_in(i,j,k+1,field_schf_re)+kr_two(i,j,k+1)*dt/2.0)
							+6.0*(sch_in(i,j,k-1,field_schf_re)+kr_two(i,j,k-1)*dt/2.0)
							+3.0*(sch_in(i+1,j+1,k,field_schf_re)+kr_two(i+1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j-1,k,field_schf_re)+kr_two(i+1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j+1,k,field_schf_re)+kr_two(i-1,j+1,k)*dt/2.0)
							+3.0*(sch_in(i-1,j-1,k,field_schf_re)+kr_two(i-1,j-1,k)*dt/2.0)
							+3.0*(sch_in(i+1,j,k+1,field_schf_re)+kr_two(i+1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i+1,j,k-1,field_schf_re)+kr_two(i+1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k+1,field_schf_re)+kr_two(i-1,j,k+1)*dt/2.0)
							+3.0*(sch_in(i-1,j,k-1,field_schf_re)+kr_two(i-1,j,k-1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k+1,field_schf_re)+kr_two(i,j+1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j+1,k-1,field_schf_re)+kr_two(i,j+1,k-1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k+1,field_schf_re)+kr_two(i,j-1,k+1)*dt/2.0)
							+3.0*(sch_in(i,j-1,k-1,field_schf_re)+kr_two(i,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_re)+kr_two(i+1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_re)+kr_two(i+1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_re)+kr_two(i+1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_re)+kr_two(i+1,j-1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_re)+kr_two(i-1,j+1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_re)+kr_two(i-1,j+1,k-1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_re)+kr_two(i-1,j-1,k+1)*dt/2.0)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_re)+kr_two(i-1,j-1,k-1)*dt/2.0)
							-88.0*(sch_in(i,j,k,field_schf_re)+kr_two(i,j,k)*dt/2.0))
			       *invdeltasq_half/52.0
#ifdef GRAVITY
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_two(i,j,k)*dt/2.0)
#endif
			       ;

			       });

	amrex::FArrayBox kr_four_fab(bxfour);
	amrex::Array4<amrex::Real> const kr_four = kr_four_fab.array();
	amrex::FArrayBox ki_four_fab(bxfour);
	amrex::Array4<amrex::Real> const ki_four = ki_four_fab.array();

	ParallelFor(bxfour,
 			   [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     kr_four(i,j,k) =-hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_im)+ki_three(i+1,j,k)*dt)
							+6.0*(sch_in(i-1,j,k,field_schf_im)+ki_three(i-1,j,k)*dt)
							+6.0*(sch_in(i,j+1,k,field_schf_im)+ki_three(i,j+1,k)*dt)
							+6.0*(sch_in(i,j-1,k,field_schf_im)+ki_three(i,j-1,k)*dt)
							+6.0*(sch_in(i,j,k+1,field_schf_im)+ki_three(i,j,k+1)*dt)
							+6.0*(sch_in(i,j,k-1,field_schf_im)+ki_three(i,j,k-1)*dt)
							+3.0*(sch_in(i+1,j+1,k,field_schf_im)+ki_three(i+1,j+1,k)*dt)
							+3.0*(sch_in(i+1,j-1,k,field_schf_im)+ki_three(i+1,j-1,k)*dt)
							+3.0*(sch_in(i-1,j+1,k,field_schf_im)+ki_three(i-1,j+1,k)*dt)
							+3.0*(sch_in(i-1,j-1,k,field_schf_im)+ki_three(i-1,j-1,k)*dt)
							+3.0*(sch_in(i+1,j,k+1,field_schf_im)+ki_three(i+1,j,k+1)*dt)
							+3.0*(sch_in(i+1,j,k-1,field_schf_im)+ki_three(i+1,j,k-1)*dt)
							+3.0*(sch_in(i-1,j,k+1,field_schf_im)+ki_three(i-1,j,k+1)*dt)
							+3.0*(sch_in(i-1,j,k-1,field_schf_im)+ki_three(i-1,j,k-1)*dt)
							+3.0*(sch_in(i,j+1,k+1,field_schf_im)+ki_three(i,j+1,k+1)*dt)
							+3.0*(sch_in(i,j+1,k-1,field_schf_im)+ki_three(i,j+1,k-1)*dt)
							+3.0*(sch_in(i,j-1,k+1,field_schf_im)+ki_three(i,j-1,k+1)*dt)
							+3.0*(sch_in(i,j-1,k-1,field_schf_im)+ki_three(i,j-1,k-1)*dt)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_im)+ki_three(i+1,j+1,k+1)*dt)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_im)+ki_three(i+1,j+1,k-1)*dt)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_im)+ki_three(i+1,j-1,k+1)*dt)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_im)+ki_three(i+1,j-1,k-1)*dt)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_im)+ki_three(i-1,j+1,k+1)*dt)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_im)+ki_three(i-1,j+1,k-1)*dt)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_im)+ki_three(i-1,j-1,k+1)*dt)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_im)+ki_three(i-1,j-1,k-1)*dt)
							-88.0*(sch_in(i,j,k,field_schf_im)+ki_three(i,j,k)*dt))
			       *invdeltasq_new/52.0
#ifdef GRAVITY
			       - phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_im)+ki_three(i,j,k)*dt)
#endif
			       ;

			     ki_four(i,j,k) = hbaroverm*(6.0*(sch_in(i+1,j,k,field_schf_re)+kr_three(i+1,j,k)*dt)
							+6.0*(sch_in(i-1,j,k,field_schf_re)+kr_three(i-1,j,k)*dt)
							+6.0*(sch_in(i,j+1,k,field_schf_re)+kr_three(i,j+1,k)*dt)
							+6.0*(sch_in(i,j-1,k,field_schf_re)+kr_three(i,j-1,k)*dt)
							+6.0*(sch_in(i,j,k+1,field_schf_re)+kr_three(i,j,k+1)*dt)
							+6.0*(sch_in(i,j,k-1,field_schf_re)+kr_three(i,j,k-1)*dt)
							+3.0*(sch_in(i+1,j+1,k,field_schf_re)+kr_three(i+1,j+1,k)*dt)
							+3.0*(sch_in(i+1,j-1,k,field_schf_re)+kr_three(i+1,j-1,k)*dt)
							+3.0*(sch_in(i-1,j+1,k,field_schf_re)+kr_three(i-1,j+1,k)*dt)
							+3.0*(sch_in(i-1,j-1,k,field_schf_re)+kr_three(i-1,j-1,k)*dt)
							+3.0*(sch_in(i+1,j,k+1,field_schf_re)+kr_three(i+1,j,k+1)*dt)
							+3.0*(sch_in(i+1,j,k-1,field_schf_re)+kr_three(i+1,j,k-1)*dt)
							+3.0*(sch_in(i-1,j,k+1,field_schf_re)+kr_three(i-1,j,k+1)*dt)
							+3.0*(sch_in(i-1,j,k-1,field_schf_re)+kr_three(i-1,j,k-1)*dt)
							+3.0*(sch_in(i,j+1,k+1,field_schf_re)+kr_three(i,j+1,k+1)*dt)
							+3.0*(sch_in(i,j+1,k-1,field_schf_re)+kr_three(i,j+1,k-1)*dt)
							+3.0*(sch_in(i,j-1,k+1,field_schf_re)+kr_three(i,j-1,k+1)*dt)
							+3.0*(sch_in(i,j-1,k-1,field_schf_re)+kr_three(i,j-1,k-1)*dt)
							+2.0*(sch_in(i+1,j+1,k+1,field_schf_re)+kr_three(i+1,j+1,k+1)*dt)
							+2.0*(sch_in(i+1,j+1,k-1,field_schf_re)+kr_three(i+1,j+1,k-1)*dt)
							+2.0*(sch_in(i+1,j-1,k+1,field_schf_re)+kr_three(i+1,j-1,k+1)*dt)
							+2.0*(sch_in(i+1,j-1,k-1,field_schf_re)+kr_three(i+1,j-1,k-1)*dt)
							+2.0*(sch_in(i-1,j+1,k+1,field_schf_re)+kr_three(i-1,j+1,k+1)*dt)
							+2.0*(sch_in(i-1,j+1,k-1,field_schf_re)+kr_three(i-1,j+1,k-1)*dt)
							+2.0*(sch_in(i-1,j-1,k+1,field_schf_re)+kr_three(i-1,j-1,k+1)*dt)
							+2.0*(sch_in(i-1,j-1,k-1,field_schf_re)+kr_three(i-1,j-1,k-1)*dt)
							-88.0*(sch_in(i,j,k,field_schf_re)+kr_three(i,j,k)*dt))
			       *invdeltasq_new/52.0
#ifdef GRAVITY
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_three(i,j,k)*dt)
#endif
			       ;

			       });

			     ParallelFor(bxfour,
 			   [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     sch_out(i,j,k,field_schf_re) = sch_in(i,j,k,field_schf_re) + dt*(kr_one(i,j,k)+2.0*kr_two(i,j,k)+2.0*kr_three(i,j,k)+kr_four(i,j,k))/6.0;			     
			     sch_out(i,j,k,field_schf_im) = sch_in(i,j,k,field_schf_im) + dt*(ki_one(i,j,k)+2.0*ki_two(i,j,k)+2.0*ki_three(i,j,k)+ki_four(i,j,k))/6.0;			     	
			     sch_out(i,j,k,AxSCH::getField(Fields::Dens)) = sch_out(i,j,k,field_schf_re)*sch_out(i,j,k,field_schf_re)+sch_out(i,j,k,field_schf_im)*sch_out(i,j,k,field_schf_im);
			     sch_out(i,j,k,AxSCH::getField(Fields::Phase)) = std::atan2(sch_out(i,j,k,field_schf_im),sch_out(i,j,k,field_schf_re));
			   });
      }

    SCH_new.FillBoundary(geom.periodicity());

#ifdef DEBUG
    if (SCH_new.contains_nan(0, SCH_new.nComp(), 0))
      {
        for (int i = 0; i < SCH_new.nComp(); i++)
          {
            if (SCH_new.contains_nan(i,1,0))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		amrex::Abort("SCH_new has NaNs in this component::advance_FDM_FD()");
              }
          }
      }
#endif


//#ifdef TEST_PLANE_WAVE
amrex::Real total_error = 0.0;
amrex::Real total_error_real = 0.0;
amrex::Real total_error_imag = 0.0;

int grid_count = 0;

// Target indices for specific output (set as needed)
const int target_i = 0; // Adjust to desired i value
const int target_j = 0; // Adjust to desired j value
const int target_k = 0; // Adjust to desired k value

for (amrex::MFIter mfi(SCH_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const amrex::Box& bx = mfi.tilebox();
    auto const& field_arr = SCH_old.array(mfi);

    const amrex::Real Lbox = geom.ProbHi(0) - geom.ProbLo(0); // Box length
	const amrex::Real SCH_k = SCH_k0 * 2. * M_PI / Lbox;
    const amrex::Real omega = SCH_k * SCH_k * hbaroverm / 2.0; // Dispersion relation

    amrex::ParallelFor(bx, [=,&total_error,&total_error_real,&total_error_imag,&grid_count] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        // Compute analytical solution
        amrex::Real x = geom.ProbLo(0) + i * geom.CellSize(0);
        amrex::Real analytic_re = SCH0 * std::cos(SCH_k * x - omega * time);
        amrex::Real analytic_im = SCH0 * std::sin(SCH_k * x - omega * time);

        // Numerical results
        amrex::Real numerical_re = field_arr(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re));
        amrex::Real numerical_im = field_arr(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im));

        // Compute errors
        amrex::Real error_re = std::abs(numerical_re - analytic_re);
        amrex::Real error_im = std::abs(numerical_im - analytic_im);
        amrex::Real cell_error = std::sqrt(error_re * error_re + error_im * error_im);

        // Update totals
        amrex::Gpu::Atomic::Add(&total_error_real, error_re);
        amrex::Gpu::Atomic::Add(&total_error_imag, error_im);
        amrex::Gpu::Atomic::Add(&total_error, cell_error);
        amrex::Gpu::Atomic::Add(&grid_count, 1);

        // Print specific grid point data
        // if (i == target_i && j == target_j && k == target_k) {
        //     amrex::Print() << "dx: " << geom.CellSize(0) << ", dt: " << dt << "\n"
						  //  << "i: " << i << ", j: " << j << ", k: " << k << ",  t=" << time << ",  x=" << x << "\n"
        //                    << "Numerical Real: " << numerical_re << ", Analytical Real: " << analytic_re << "\n"
        //                    << "Numerical Imag: " << numerical_im << ", Analytical Imag: " << analytic_im << "\n";
        // }
    });
}

// Output results
// if (amrex::ParallelDescriptor::IOProcessor()) {
//     amrex::Print() << "Average Real Error: " << (total_error_real / grid_count) << "\n";
//     amrex::Print() << "Average Imag Error: " << (total_error_imag / grid_count) << "\n";
//     amrex::Print() << "Average Error: " << (total_error / grid_count) << "\n";
// }
//#endif




    // Only advance the scale-factor with the root grid
 /*    if(level == 0)
    
        Comoving::kick_a(dt_half, true); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
        Comoving::drift_a(dt);
        Comoving::reset_rho();
        fill_rho();
        Comoving::kick_a(dt_half, false); // See Comoving_Full.cpp. The calculation of the acceleration for the scale factor is a little bit complicated because it involves the first derivative, ap.
 */
    BL_PROFILE_VAR_STOP(SCH_ADVANCE);
    
    return dt;
}

