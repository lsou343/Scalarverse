#include <AxSCHComov.H>
#include <Comoving_EOS.H>
#include <AMReX_MultiFabUtil.H>

#ifdef GRAV
#include <AxSP.H>
#endif

using namespace amrex;

void FindMaxPsiRealImag(MultiFab &Ax_new) {
  Real max_re_host = -1e30; // Host variable for max real part
  IntVect max_re_idx_host;

  for (MFIter mfi(Ax_new, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrPsi = Ax_new.array(mfi);

    // Allocate a GPU scalar for atomic max tracking
    Gpu::DeviceScalar<Real> max_re_gpu(-1e30);
    Real *max_re_ptr = max_re_gpu.dataPtr();

    // Allocate device storage for index tracking
    Gpu::DeviceVector<int> max_idx_gpu(3, -1);
    int *max_idx_ptr = max_idx_gpu.dataPtr();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Real psi_val = arrPsi(i, j, k);
      Real re = arrPsi(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re));

      // Atomic max update
      if (re > *max_re_ptr) {
        amrex::Gpu::Atomic::Max(max_re_ptr, re);
        max_idx_ptr[0] = i;
        max_idx_ptr[1] = j;
        max_idx_ptr[2] = k;
      }
    });

    // Copy results back to host
    Gpu::copy(Gpu::deviceToHost, max_re_ptr, max_re_ptr + 1, &max_re_host);
    Gpu::copy(Gpu::deviceToHost, max_idx_ptr, max_idx_ptr + 3,
              max_re_idx_host.getVect());

    // Synchronize GPU to ensure correctness
    Gpu::streamSynchronize();
  }

  // Print results after computation
  amrex::Print() << "Max Real: " << max_re_host << " at " << max_re_idx_host
  << "\n";
}


void AxSCHComov::advance_SCH_FD(Real time,
              Real dt,
              Real a_old,
              Real a_new)
{
   
	// // calcualte scale factors
 //    const Real a_old = Comoving::get_comoving_a(time);
    const Real a_half = Comoving::get_comoving_a(time+0.5*dt);
 //    const Real a_new = Comoving::get_comoving_a(time+dt);
  	// Print() << "a_old: " << a_old << '\n';
 //    const Real a_old = 1;
 //    const Real a_half = 1;
 //    const Real a_new = 1;
	// 
		// Print get_comiving_a(time) to check if the function is working
		// Print() << "a_test: " << Comoving::get_comoving_a(time) << '\n';
		// Print() << "a_new: " << Comoving::get_comoving_a(time+dt) << '\n'; 

    // define cell size dx
    const Real* dx      = geom.CellSize();


    // calculate 1/a^2dx for different times
    const Real invdeltasq_old  = 1.0 / ( a_old  * dx[0] ) / ( a_old  * dx[0] );                                                                                                       
    const Real invdeltasq_half = 1.0 / ( a_half * dx[0] ) / ( a_half * dx[0] );                                                                                                       
    const Real invdeltasq_new  = 1.0 / ( a_new  * dx[0] ) / ( a_new  * dx[0] );


	const Real dt_half = 0.5*dt;
	  MultiFab&  SCH_old = get_level(level).get_old_data(getState(StateType::SCH_Type));
    // print SCH_old
    // print example Print() << "Initializing the data at level " << level << '\n';
   // Print() << "SCH_old: " << SCH_old.boxArray() << '\n';
    //IntVect cell{0, 0, 0};
   //  Print() << "before initializing SCH_old" << '\n';
  	// FindMaxPsiRealImag(SCH_old); 

    //print_state(SCH_old,cell);

    MultiFab&  SCH_new = get_level(level).get_new_data(getState(StateType::SCH_Type));   
    //print_state(SCH_new,cell);
  //   Print() << "after initializing SCH_new" << '\n';
		// FindMaxPsiRealImag(SCH_new);

#ifdef DEBUG
    if (SCH_old.contains_nan(0, SCH_old.nComp(), 0))
      {
        for (int i = 0; i < SCH_old.nComp(); i++)
          {
            if (SCH_old.contains_nan(i,1,0))
              {
		std::cout << "Testing component i for NaNs: " << i << std::endl;
		Abort("SCH_old has NaNs in this component::AxSCHComov advance()");
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

#ifdef GRAV
	    MultiFab& Phi_old = get_old_data(AxSP::getState(AxSP::StateType::PhiGrav_Type));
	      for (FillPatchIterator
	   fpi(*this, SCH_old, 4, time, AxSCH::getState(AxSCH::StateType::SCH_Type),   0, AxSCH::nFields()),

	   pfpi(*this, Phi_old, 3, time, AxSP::getState(AxSP::StateType::PhiGrav_Type), 0, 1);
	 fpi.isValid() && pfpi.isValid(); ++fpi,++pfpi)
#else
    for (FillPatchIterator
	   fpi(*this, SCH_old, 4, time, getState(StateType::SCH_Type),   0, AxSCHComov::nFields());
	 fpi.isValid(); ++fpi)
#endif
      {
	const Box& bxfour  = fpi.validbox();
	const Box& bxthree = grow(bxfour,1);
	const Box& bxtwo   = grow(bxthree,1);
	const Box& bxone   = grow(bxtwo,1);
	auto const sch_in   = fpi().array();
	auto const sch_out  = SCH_new[fpi].array();

#ifdef GRAV
	auto const phi     = pfpi().array();
#endif
	FArrayBox kr_one_fab(bxone);
	Array4<amrex::Real> const kr_one = kr_one_fab.array();
	FArrayBox ki_one_fab(bxone);
	Array4<amrex::Real> const ki_one = ki_one_fab.array();
	//how to call AxSCHComov::getField(Fields::SCHf_Im)_im outside of the following loop to make it a variable?
	const int field_schf_im = AxSCHComov::getField(Fields::SCHf_Im);
	const int field_schf_re = AxSCHComov::getField(Fields::SCHf_Re);

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
#ifdef GRAV
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
#ifdef GRAV
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re))
#endif
			       ;
			       
			      });

	FArrayBox kr_two_fab(bxtwo);
	Array4<amrex::Real> const kr_two = kr_two_fab.array();
	FArrayBox ki_two_fab(bxtwo);
	Array4<amrex::Real> const ki_two = ki_two_fab.array();

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
#ifdef GRAV
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
#ifdef GRAV
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_one(i,j,k)*dt/2.0)
#endif
			       ;

			       });

	FArrayBox kr_three_fab(bxthree);
	Array4<amrex::Real> const kr_three = kr_three_fab.array();
	FArrayBox ki_three_fab(bxthree);
	Array4<amrex::Real> const ki_three = ki_three_fab.array();

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
#ifdef GRAV
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
#ifdef GRAV
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_two(i,j,k)*dt/2.0)
#endif
			       ;

			       });

	FArrayBox kr_four_fab(bxfour);
	Array4<amrex::Real> const kr_four = kr_four_fab.array();
	FArrayBox ki_four_fab(bxfour);
	Array4<amrex::Real> const ki_four = ki_four_fab.array();

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
#ifdef GRAV
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
#ifdef GRAV
			       + phi(i,j,k)/hbaroverm*(sch_in(i,j,k,field_schf_re)+kr_three(i,j,k)*dt)
#endif
			       ;

			       });

			     ParallelFor(bxfour,
 			   [=] AMREX_GPU_DEVICE (int i, int j, int k)
			   {
			     sch_out(i,j,k,field_schf_re) = sch_in(i,j,k,field_schf_re) + dt*(kr_one(i,j,k)+2.0*kr_two(i,j,k)+2.0*kr_three(i,j,k)+kr_four(i,j,k))/6.0;			     
			     sch_out(i,j,k,field_schf_im) = sch_in(i,j,k,field_schf_im) + dt*(ki_one(i,j,k)+2.0*ki_two(i,j,k)+2.0*ki_three(i,j,k)+ki_four(i,j,k))/6.0;			     	
			     sch_out(i,j,k,AxSCHComov::getField(Fields::Dens)) = sch_out(i,j,k,field_schf_re)*sch_out(i,j,k,field_schf_re)+sch_out(i,j,k,field_schf_im)*sch_out(i,j,k,field_schf_im);
			     sch_out(i,j,k,AxSCHComov::getField(Fields::Phase)) = std::atan2(sch_out(i,j,k,field_schf_im),sch_out(i,j,k,field_schf_re));
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
		Abort("SCH_new has NaNs in this component::advance_FDM_FD()");
              }
          }
      }
#endif
//
// //
// Real total_error = 0.0;
// Real total_error_real = 0.0;
// Real total_error_imag = 0.0;
//
// int grid_count = 0;
//
// // Target indices for specific output (set as needed)
// const int target_i = 0; //1; // Adjust to desired i value
// const int target_j = 0; //1; // Adjust to desired j value
// const int target_k = 0; //1; // Adjust to desired k value
//
// // for (MFIter mfi(SCH_old, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
// for (MFIter mfi(SCH_old, false); mfi.isValid(); ++mfi) {
//     // const Box& bx = mfi.tilebox();
// 		const Box& bx = mfi.validbox();
//     auto const& field_arr = SCH_old.array(mfi);
//
//     const Real Lbox = geom.ProbHi(0) - geom.ProbLo(0); // Box length
// 	const Real SCH_k = SCH_k0 * 2. * M_PI / Lbox;
//     const Real omega = SCH_k * SCH_k * hbaroverm / 2.0; // Dispersion relation
//
//     ParallelFor(bx, [=,&total_error,&total_error_real,&total_error_imag,&grid_count] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
//         // Compute analytical solution
//         Real x = geom.ProbLo(0) + i * geom.CellSize(0);
//         Real analytic_re = SCH0 * std::cos(SCH_k * x - omega * time);
//         Real analytic_im = SCH0 * std::sin(SCH_k * x - omega * time);
//
//         // Numerical results
//         Real numerical_re = field_arr(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re));
//         Real numerical_im = field_arr(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Im));
//
//         // Compute errors
//         Real error_re = std::abs(numerical_re - analytic_re);
//         Real error_im = std::abs(numerical_im - analytic_im);
//         Real cell_error = std::sqrt(error_re * error_re + error_im * error_im);
//
//         // Update totals
//         Gpu::Atomic::Add(&total_error_real, error_re);
//         Gpu::Atomic::Add(&total_error_imag, error_im);
//         Gpu::Atomic::Add(&total_error, cell_error);
//         Gpu::Atomic::Add(&grid_count, 1);
//         // Print specific grid point data
//         if (i == target_i && j == target_j && k == target_k) {
//             Print() << "dx: " << geom.CellSize(0) << ", dt: " << dt << "\n"
// 						   << "i: " << i << ", j: " << j << ", k: " << k << ",  t=" << time << ",  x=" << x << "\n"
//                            << "Numerical Real: " << numerical_re << ", Analytical Real: " << analytic_re << "\n"
//                            << "Numerical Imag: " << numerical_im << ", Analytical Imag: " << analytic_im << "\n";
//         }
//     });
// }
//
// // Output results
// if (ParallelDescriptor::IOProcessor()) {
//     Print() << "Average Real Error: " << (total_error_real / grid_count) << "\n";
//     Print() << "Average Imag Error: " << (total_error_imag / grid_count) << "\n";
//     Print() << "Average Error: " << (total_error / grid_count) << "\n";
// }

    BL_PROFILE_VAR_STOP(SCHCOMOV_ADVANCE);
}

