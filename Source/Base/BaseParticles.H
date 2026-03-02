//////////////////////////////////////////////////////////////////////////
//
//   BaseParticles.cpp
//
/////////////////////////////////////////////////////////////////////////

void BaseAx::post_timestep (int iteration)
{
    BL_PROFILE("BaseAx::post_timestep()");

    amrex::Gpu::LaunchSafeGuard lsg(true);
    MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

    //
    // Integration cycle on fine level grids is complete
    // do post_timestep stuff here.
    //
    int finest_level = parent->finestLevel();
    const int ncycle = parent->nCycle(level);

    BL_PROFILE_VAR("BaseAx::post_timestep()::remove_virt_ghost",rm);
    //
    // Remove virtual particles at this level if we have any.
    //
    remove_virtual_particles();

    //
    // Remove Ghost particles on the final iteration
    //
    if (iteration == ncycle)
        remove_ghost_particles();
    amrex::Gpu::streamSynchronize();
    BL_PROFILE_VAR_STOP(rm);
    BL_PROFILE_VAR("Nyx::post_timestep()::redist",redist);

    if(load_balance_int < 0 || !(nStep() % load_balance_int == 0 && level == 0 && (new_a >= 1.0/(load_balance_start_z + 1.0))))
    {

    //
    // Sync up if we're level 0 or if we have particles that may have moved
    // off the next finest level and need to be added to our own level.
    //
    if ((iteration < ncycle and level < finest_level) || level == 0)
    {
        for (int i = 0; i < theActiveParticles().size(); i++)
        {
	  if(finest_level == 0){
                 theActiveParticles()[i]->RedistributeLocal(level,
                                                  theActiveParticles()[i]->finestLevel(),
                                                  iteration);
#ifdef FDM
		 if(theFDMphasePC())
		   theFDMphasePC()->RedistributeLocal(level,
						 theFDMphasePC()->finestLevel(),
						 iteration);
#endif
	  } else {
                 theActiveParticles()[i]->Redistribute(level,
                                                  theActiveParticles()[i]->finestLevel(),
                                                  iteration);
#ifdef FDM
		 if(theFDMphasePC())
		   theFDMphasePC()->Redistribute(level,
						 theFDMphasePC()->finestLevel(),
						 iteration);
#endif
	  }
	  if(shrink_to_fit){
                 theActiveParticles()[i]->ShrinkToFit();
#ifdef FDM
		 theFDMphasePC()->ShrinkToFit();
#endif
	  }
        }
    }

    }

}

void BaseAx::post_restart ()
{
    BL_PROFILE("BaseAx::post_restart()");
    if (level == 0)
        particle_post_restart(parent->theRestartFile());

}

void BaseAx::postCoarseTimeStep (Real cumtime)
{
   BL_PROFILE("BaseAx::postCoarseTimeStep()");
   MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

    if(load_balance_int >= 0 && nStep() % load_balance_int == 0 && (new_a >= 1.0/(load_balance_start_z + 1.0)))
    {
      if(verbose>0)
        amrex::Print()<<"Load balancing since "<<nStep()<<" mod "<<load_balance_int<<" == 0"<<std::endl;

    for (int lev = 0; lev <= parent->finestLevel(); lev++)
    {
    
        Vector<long> wgts(grids.size());
        DistributionMapping dm;

        //Should these weights be constructed based on level=0, or lev from for?
        if(load_balance_wgt_strategy == 0)
        {
            for (unsigned int i = 0; i < wgts.size(); i++)
            {
                wgts[i] = grids[i].numPts();
            }
            if(load_balance_strategy==DistributionMapping::Strategy::KNAPSACK)
                dm.KnapSackProcessorMap(wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::SFC)
                dm.SFCProcessorMap(grids, wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::ROUNDROBIN)
                dm.RoundRobinProcessorMap(wgts, load_balance_wgt_nmax);
        }
        else if(load_balance_wgt_strategy == 1)
        {
#ifdef FDM
	  /*FDMphase particles need more computions than DM particles so we should load balance them. */
	    wgts = theFDMphasePC()->NumberOfParticlesInGrid(level,false,false);
#else
            wgts = theDMPC()->NumberOfParticlesInGrid(level,false,false);
#endif
            if(load_balance_strategy==DistributionMapping::Strategy::KNAPSACK)
                dm.KnapSackProcessorMap(wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::SFC)
                dm.SFCProcessorMap(grids, wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::ROUNDROBIN)
                dm.RoundRobinProcessorMap(wgts, load_balance_wgt_nmax);
        }
        else if(load_balance_wgt_strategy == 2)
        {
            MultiFab particle_mf(grids,theDMPC()->ParticleDistributionMap(level),1,1);
            theDMPC()->Increment(particle_mf, level);
            if(load_balance_strategy==DistributionMapping::Strategy::KNAPSACK)
                dm.KnapSackProcessorMap(wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::SFC)
                dm.SFCProcessorMap(grids, wgts, load_balance_wgt_nmax);
            else if(load_balance_strategy==DistributionMapping::Strategy::ROUNDROBIN)
                dm.RoundRobinProcessorMap(wgts, load_balance_wgt_nmax);
        }
        else
        {
            amrex::Abort("Selected load balancing strategy not implemented");
        }

        amrex::Gpu::Device::streamSynchronize();
        const DistributionMapping& newdmap = dm;
        
        for (int i = 0; i < theActiveParticles().size(); i++)
        {
             theActiveParticles()[i]->Regrid(newdmap, grids, lev);

             if(shrink_to_fit)
                 theActiveParticles()[i]->ShrinkToFit();
        }

    amrex::Gpu::streamSynchronize();
    }

    }

   AmrLevel::postCoarseTimeStep(cumtime);

    //
    // postCoarseTimeStep() is only called by level 0.
    //
    if (Nyx::theDMPC() && particle_move_type == "Random")
        particle_move_random();

   int nstep = parent->levelSteps(0);

   if(verbose>1)
   {
   amrex::Print()<<"End of postCoarseTimeStep, printing:"<<std::endl;
   MultiFab::printMemUsage();
   amrex::Arena::PrintUsage();
   }

}

void BaseAx::post_regrid (int lbase,
                  int new_finest)
{
    BL_PROFILE("Nyx::post_regrid()");
    if (level == lbase) {
        amrex::Gpu::LaunchSafeGuard lsg(true);
        particle_redistribute(lbase, false);
    }
    amrex::Gpu::Device::streamSynchronize();

/* #ifdef GRAVITY

    int which_level_being_advanced = parent->level_being_advanced();

    bool do_grav_solve_here;

    if (which_level_being_advanced >= 0)
    {
        do_grav_solve_here = (level == which_level_being_advanced) && (lbase == which_level_being_advanced);
    } else {
        do_grav_solve_here = (level == lbase);
    }

    // Only do solve here if we will be using it in the timestep right after without re-solving,
    //      or if this is called from somewhere other than Amr::timeStep
    const Real cur_time = state[State_for_Time].curTime();
    if (do_grav && (cur_time > 0) && do_grav_solve_here)
    {
        int ngrow_for_solve = parent->levelCount(level) + 1;
        int use_previous_phi_as_guess = 1;
        gravity->multilevel_solve_for_new_phi(level, new_finest, ngrow_for_solve, use_previous_phi_as_guess);
    }
#endif
    delete fine_mask;
    fine_mask = 0; */
}

void BaseAx::CreateLevelDirectory (const std::string &dir)
{
    // AmrLevel::CreateLevelDirectory(dir);  // ---- this sets levelDirectoryCreated = true

    std::string LevelDir, FullPath;
    if(theDMPC()){
        std::string dm(dir + "/" + Nyx::retrieveDM());
        if(ParallelDescriptor::IOProcessor()) {
            if( ! amrex::UtilCreateDirectory(dm, 0755)) {
                amrex::CreateDirectoryFailed(dm);
            }
        }

        LevelDirectoryNames(dir, Nyx::retrieveDM(), LevelDir, FullPath);
        if(ParallelDescriptor::IOProcessor()) {
            if( ! amrex::UtilCreateDirectory(FullPath, 0755)) {
                amrex::CreateDirectoryFailed(FullPath);
            }
        }
    }

}
//////////////////////////////////////////////////////////////////////////
//
//   BaseParticles.H
//
/////////////////////////////////////////////////////////////////////////

		static std::string particle_plotfile_format;
		// How do we want to initialize the particles?
		// Must be "Random", "Cosmological" or "AsciiFile"
		static std::string particle_init_type;

		// How do we want to move the particles?
		// Must be "Random" or "Gravitational"
		static std::string particle_move_type;

		// These control random initialization
		static bool particle_initrandom_serialize;
		static long particle_initrandom_count;
		static long particle_initrandom_count_per_box;
		static amrex::Real particle_initrandom_mass;
		static int particle_initrandom_iseed;
		static int particle_skip_factor;
		static amrex::Real particle_inituniform_mass;
		static amrex::Real particle_inituniform_vx;
		static amrex::Real particle_inituniform_vy;
		static amrex::Real particle_inituniform_vz;

		// This controls the LaunchSafeGuard for particle initialization
		static int particle_launch_ics;

		//
		// Read particle-related inputs
		//
		static void read_particle_params();

		//
		// Write particles in checkpoint directories
		//
		void particle_check_point(const std::string& dir);

		//
		// Write particles in plotfile directories
		//
		void particle_plot_file(const std::string& dir);

		//
		// How to initialize at restart
		//
		void particle_post_restart(const std::string& restart_file, bool is_checkpoint = true);

		//
		// Redistribute
		//
		void particle_redistribute(int lbase = 0, bool init = false);

		//
		// Move randomly
		//
		void particle_move_random();

		//
		// Initialize particle locations and velocities (and strengths if relevant)
		//
		virtual void init_particles();

		//
		// Setup virtual particles if necessary
		//
		void setup_virtual_particles();

		//
		// Remove virtual particles if necessary
		//
		void remove_virtual_particles();
		//
		// Setup ghost particles (for finer levels) if necessary
		//
		void setup_ghost_particles(int ngrow);

		//
		// Remove ghost particles (for this level) if necessary
		//
		void remove_ghost_particles();

		//
		// Time step control based on particles
		//
		void particle_est_time_step(amrex::Real& est_dt);

		//
		// Derived quantities associated with particles
		//
		std::unique_ptr<amrex::MultiFab> particle_derive (const std::string& name, amrex::Real time, int ngrow);

		//
		// Default verbosity of Particle class
		//
		static int particle_verbose;
        
		//
		// Shall we write the initial single-level particle density into a multifab
		//   called "ParticleDensity"?
		//
		static int write_particle_density_at_init;

		static amrex::Vector<NyxParticleContainerBase*>& theActiveParticles();
		static amrex::Vector<NyxParticleContainerBase*>& theVirtualParticles();
		static amrex::Vector<NyxParticleContainerBase*>& theGhostParticles();

		static DarkMatterParticleContainer* theDMPC();
		static DarkMatterParticleContainer* theVirtPC();
		static DarkMatterParticleContainer* theGhostPC();

		static StellarParticleContainer* theSPC();
		static StellarParticleContainer* theVirtSPC();
		static StellarParticleContainer* theGhostSPC();
