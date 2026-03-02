#include <AxKG.H>

#include <bc_fill.H>

#include <KGDerive.H>

// #include <Prob.H>

namespace
{

    // We should find a better place for these two.
    int scalar_bc[] =
    {
        // INT_DIR, EXT_DIR, FOEXTRAP, REFLECT_EVEN, REFLECT_EVEN, REFLECT_EVEN
         amrex::BCType::int_dir, amrex::BCType::ext_dir, amrex::BCType::foextrap, amrex::BCType::reflect_even, amrex::BCType::reflect_even, amrex::BCType::reflect_even
    };


    void set_scalar_bc(amrex::BCRec& bc, const amrex::BCRec& phys_bc)
    {
        const int* lo_bc = phys_bc.lo();
        const int* hi_bc = phys_bc.hi();
        for (int i = 0; i < BL_SPACEDIM; i++)
        {
            bc.setLo(i, scalar_bc[lo_bc[i]]);
            bc.setHi(i, scalar_bc[hi_bc[i]]);
        }
    }
    

}

int BaseAx::NUM_STATE = AxKG::nStates();       // This must be declared in the derived class

AxKG::AxKG ()
{
    BL_PROFILE("AxKG::AxKG()");
    fine_mask = 0;  
}

AxKG::AxKG (amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
            amrex::Real time)
    :
    BaseAx(papa,lev,level_geom,bl,dm,time)
{
    BL_PROFILE("AxKG::AxKG(Amr)");
}

void AxKG::init (AmrLevel& old)
{
    BaseAx::init(old);

    amrex::MultiFab&  KG_new = get_new_data(getState(StateType::KG_Type));

    AxKG* old_level = static_cast<AxKG*> (&old);
    amrex::Real cur_time  = old_level->state[State_for_Time].curTime();

	// In Amr/AMReX_AmrLevel.H  --PH
    /* static void FillPatch (AmrLevel& amrlevel, */
    /*                        MultiFab& leveldata, */
    /*                        int       boxGrow, */
    /*                        Real      time, */
    /*                        int       index, */
    /*                        int       scomp, */
    /*                        int       ncomp, */
    /*                        int       dcomp=0); */
    FillPatch(old, KG_new, 0, cur_time, getState(StateType::KG_Type), 0, nFields());

    amrex::Gpu::Device::streamSynchronize();
}

//
// This version inits the data on a new level that did not
// exist before regridding.
//
void AxKG::init ()
{
    BaseAx::init();

    amrex::Real cur_time  = static_cast<AxKG*>(&get_level(level-1))->state[State_for_Time].curTime();

    amrex::MultiFab&  Ax_new = get_new_data(getState(StateType::KG_Type));
    FillCoarsePatch(Ax_new, 0, cur_time, getState(StateType::KG_Type), 0, Ax_new.nComp());
    
    // We set dt to be large for this new level to avoid screwing up
    // computeNewDt.
    parent->setDtLevel(1.e100, level);
}

void AxKG::initData()
{
	BL_PROFILE("AxKG::initData()");

	amrex::Gpu::LaunchSafeGuard lsg(true);
	// Here we initialize the grid data and the particles from a plotfile.
	if (!parent->theRestartPlotFile().empty())
	{
		amrex::Abort("AmrData requires fortran");  // This is a bizarre error message...
		return;
	}

	if (verbose && amrex::ParallelDescriptor::IOProcessor())
		amrex::Print() << "Initializing the data at level " << level << '\n';

	const auto dx = geom.CellSizeArray();
	const auto geomdata = geom.data();

	// Make sure dx = dy = dz -- that's all we guarantee to support
	const amrex::Real SMALL = 1.e-13;
	if ( (fabs(dx[0] - dx[1]) > SMALL) || (fabs(dx[0] - dx[2]) > SMALL) )
		amrex::Abort("We don't support dx != dy != dz");

	amrex::MultiFab& KG_new = get_new_data(getState(StateType::KG_Type));
    amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> prob_param; // Array of parameters required for initial conditions.
    prob_param.fill(PAR_ERR_VAL); // Fill it with error values first.
    prob_param_fill(prob_param); // Add what values are needed.

    // We have to do things slightly differently if we're initializing in position-space
    // or Fourier space.
    if(ic == ICType::uniform || ic == ICType::fixed_k)
    {
        for (amrex::MFIter mfi(KG_new,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox();
            const auto fab_KG_new=KG_new.array(mfi);
            prob_initdata_pos_on_box(bx, fab_KG_new, geomdata, prob_param); // This function is defined in Prob.H. --PH
        }
        KG_new.FillBoundary(geom.periodicity());
    }
    else
    {
#ifdef BL_USE_MPI
	printf("\n\nTest1\n\n");
        prob_initdata_mom_on_mf(KG_new, geom, prob_param);
        printf("\n\nTest2\n\n");
        KG_new.plus(1., 0, 1);  // (val, comp, ncomp): Adds the value val to ncomp components starting at comp. Note f_pr starts at 1 and f_pr = f/KG0.
        KG_new.FillBoundary(geom.periodicity());
#endif
    }
}

amrex::Real AxKG::est_time_step (amrex::Real dt_old)
{
    BL_PROFILE("AxKG::est_time_step()");

    // Currently the only option
    if (BaseAx::fixed_dt > 0)
        return BaseAx::fixed_dt;

    return 0;
}

void AxKG::average_down ()
{
    // This is only implemented in the derived class because we have to specify the state type.
    // However, for hydro code there is more to be done, and an overwrite of average_down(state_type) is in order.  --PH

    BL_PROFILE("AxKG::average_down()");
    if (level == parent->finestLevel()) return;

    BaseAx::average_down(getState(StateType::KG_Type));
}

void AxKG::variable_setup()
{
    BaseAx::variable_setup();
    
    // Get options, set phys_bc  --- This is necessary because these are all static functions, so there's no actual inheritance!
    AxKG::read_params();
    
    // Note that the default is state_data_extrap = false,
    // store_in_checkpoint = true.  We only need to put these in
    // explicitly if we want to do something different,
    // like not store the state data in a checkpoint directory
    bool state_data_extrap = false;
    bool store_in_checkpoint = true;

    amrex::BCRec bc;

    amrex::StateDescriptor::BndryFunc bndryfunc(nyx_bcfill);
    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.

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
    
    amrex::Interpolater* KGinterp;
    KGinterp  = &amrex::cell_bilinear_interp;

    // Establish the workhorse fields
    desc_lst.addDescriptor(getState(StateType::KG_Type), amrex::IndexType::TheCellType(),
                           amrex::StateDescriptor::Point, 0, nFields(), 
			   KGinterp, state_data_extrap, 
			   store_in_checkpoint);

    set_scalar_bc(bc, phys_bc);

    desc_lst.setComponent(getState(StateType::KG_Type), 0, "KGfpr", bc,
                          bndryfunc);
    desc_lst.setComponent(getState(StateType::KG_Type), 1, "KGfVpr", bc,
                          bndryfunc);  // Track the field and its derivative for leapfrog integration --PH

    // Establish the derived fields
    derive_lst.add("KGf", amrex::IndexType::TheCellType(), 1, Derived::derKGf, Derived::grow_box_by_one);
    derive_lst.addComponent("KGf", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);

    derive_lst.add("KGfv", amrex::IndexType::TheCellType(), 1, Derived::derKGfv, Derived::grow_box_by_one);
    // The original field derivative requires both the program field and its time-derivative.
    derive_lst.addComponent("KGfv", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);
    derive_lst.addComponent("KGfv", desc_lst, getState(StateType::KG_Type), getField(Fields::KGfv), 1);

    derive_lst.add("KGfdens", amrex::IndexType::TheCellType(), 1, Derived::derKGfdens, Derived::grow_box_by_one);
    derive_lst.addComponent("KGfdens", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);

    // Energy density (in real units)
    derive_lst.add("Edens", amrex::IndexType::TheCellType(), 1, Derived::derEdens, Derived::grow_box_by_one);
    derive_lst.addComponent("Edens", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);
    derive_lst.addComponent("Edens", desc_lst, getState(StateType::KG_Type), getField(Fields::KGfv), 1);
    derive_lst.add("Egrad", amrex::IndexType::TheCellType(), 1, Derived::derEgrad, Derived::grow_box_by_one);
    derive_lst.addComponent("Egrad", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);
    derive_lst.addComponent("Egrad", desc_lst, getState(StateType::KG_Type), getField(Fields::KGfv), 1);
    derive_lst.add("Epot", amrex::IndexType::TheCellType(), 1, Derived::derEpot, Derived::grow_box_by_one);
    derive_lst.addComponent("Epot", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);
    derive_lst.addComponent("Epot", desc_lst, getState(StateType::KG_Type), getField(Fields::KGfv), 1);
    derive_lst.add("Ekin", amrex::IndexType::TheCellType(), 1, Derived::derEkin, Derived::grow_box_by_one);
    derive_lst.addComponent("Ekin", desc_lst, getState(StateType::KG_Type), getField(Fields::KGf), 1);
    derive_lst.addComponent("Ekin", desc_lst, getState(StateType::KG_Type), getField(Fields::KGfv), 1);

}

// Helper functions to map fields and states. This will be very useful when combining different types of simulations (e.g., KG, gravity, particles, etc.)
int AxKG::nFields()
{
    // We have two fields right now, // LSR -- this all seems weird. Works for now but makes adding new fields very complicated.
    // KGf
    // KGfv
    return 2; 
}
int AxKG::getField(Fields f)
{
    switch(f)
    {
        case Fields::KGf:
            return 0;
        case Fields::KGfv:
            return 1;
    }
    return -1;
}
AxKG::Fields AxKG::getField(int f)
{
    switch(f)
    {
        case 0:
            return Fields::KGf;
        case 1:
            return Fields::KGfv;
    }
    return Fields::KGf; // TODO: This should be an error value.
}
int AxKG::nStates()
{
    return 1; 
}
int AxKG::getState(StateType st)
{
    switch(st)
    {
        // NOTE: This must be the state used to track the time variable.
        case StateType::KG_Type:
            return 0;
    }
    return -1;
}
AxKG::StateType AxKG::getState(int st)
{
    switch(st)
    {
        // NOTE: This must be the state used to track the time variable.
        case 0:
            return StateType::KG_Type;
    }
    return StateType::KG_Type; // TODO: This should be an error value.
}
int AxKG::getIC(ICType it)
{
    switch(it)
    {
        case ICType::uniform:
            return 0;
        case ICType::fixed_k:
            return 1;
        case ICType::delta_k:
            return 2;
        case ICType::standard:
            return 3;
    }
    return -1;
}
AxKG::ICType AxKG::getIC(int it)
{
    switch(it)
    {
        case 0:
            return ICType::uniform;
        case 1:
            return ICType::fixed_k;
        case 2:
            return ICType::delta_k;
        case 3:
            return ICType::standard;
    }

    return ICType::uniform; // TODO: This should be an error value.
}
void AxKG::prob_param_fill(amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_params)
{
    prob_params[0] = getIC(ic);     

    switch (ic)
    {
        case ICType::uniform:
            prob_params[1] = KG0;
            break;
        case ICType::fixed_k:
            prob_params[1] = KG0;
            prob_params[2] = KG_k;
            break;
        case ICType::delta_k:
            prob_params[1] = KG0;
            prob_params[2] = KG_k;
            break;
        case ICType::standard:
            prob_params[1] = KG0;
            prob_params[2] = A;
            prob_params[3] = B;
            prob_params[4] = cutoff_k;

            break;
    }
}

// void AxKG::prob_initdata_mom_on_mf(amrex::MultiFab &mf ,
//                                 amrex::Geometry const& geom,
//                                 const amrex::GpuArray<amrex::Real,BaseAx::max_prob_param>& prob_param)
// {
//          // In principle this shouldn't be necessary anymore.
//     // We have real fields in position space, so we need to enforce f_k = f_{-k}, which cannot be done
//     // in BaseAx (nor should it be), so we need to re-implement this here.

//     BL_PROFILE("AxKG::prob_initdata_mom_on_mf")

//     // const auto geomdata = geom.data();

//     // amrex::MultiFab fillK(mf.boxArray(), mf.DistributionMap(), mf.nComp()*2, 0); 
//     amrex::FFT::R2C fft(geom.Domain());
//     auto const& [ba,dm] = fft.getSpectralDataLayout();
//     amrex::cMultiFab fillK(ba, dm, mf.nComp(), 0);

//     // Initialize the Fourier data
//     for (amrex::MFIter mfi(fillK,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
//     {
//         const amrex::Box& bx  = mfi.tilebox();
//         const auto fab_new = fillK.array(mfi);

//         amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) noexcept  
//         // NOTE THE & instead of =!! This takes references for external variables as opposed to making local copies.
//         {
//             prob_initdata_mom(i, j ,k, fab_new, geom.data(), prob_param);

//             // Set the 0-mode to 0 since we add it in later
//             if(i == 0 && j == 0 && k == 0)
//             {
//                 for(int comp = 0; comp < fillK.nComp(); comp++)
//                     fab_new(0, 0, 0, comp) = amrex::GpuComplex(0.,0.);  // Set the 0-mode to 0 since we add it in later
//             }
//         });
//     }

//     amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::backward> fft_backward(geom.Domain());
//     fft_backward.backward(fillK, mf);
//     mf.mult(1./mf.boxArray().numPts());
//     mf.FillBoundary(geom.periodicity());  
// }

void AxKG::errorEst (amrex::TagBoxArray& tags,
               int          clearval,
               int          tagval,
               amrex::Real         time,
               int          n_error_buf,
               int          ngrow)
{
    BL_PROFILE("BaseAx::errorEst()");

    for (int j=0; j<errtags.size(); ++j) 
    {
        std::unique_ptr<amrex::MultiFab> mf;
        if (errtags[0].Field() != std::string()) {

            // Can't seem to make relative energy density into a derived field (needs the full MultiFab, but derived field calculations only get FABs)
            if(errtags[0].Field() == "EdensRel")
            {
                mf = std::unique_ptr<amrex::MultiFab>(derive("Edens", time, errtags[j].NGrow()));

                int gridsize = get_level(0).Geom().Domain().length(0)*geom.Domain().length(1)*geom.Domain().length(2);
                amrex::Real avE = static_cast<AxKG*>(&get_level(0))->derive("Edens", time, errtags[j].NGrow())->sum()/gridsize; // Need to use level 0 to calculate the average
                mf->mult(1./avE);
            }
            else
            {
                mf = std::unique_ptr<amrex::MultiFab>(derive(errtags[j].Field(), time, errtags[j].NGrow()));
            }
        }
        errtags[j](tags,mf.get(),clearval,tagval,time,level,geom);
    }
}
