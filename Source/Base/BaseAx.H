#ifndef _BASE_AX_H_
#define _BASE_AX_H_

// AMReX
#include <AMReX_AmrLevel.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_ErrorList.H>
#include <AMReX_FFT.H> // Now built-in
#include <AMReX_FluxRegister.H>
#include <AMReX_LevelBld.H>
#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>

// C++
#include <iomanip>
#include <iostream>
#include <sstream>

// We have a bit of an odd system at the moment for passing parameter values to
// the initial condition machine, Prob.H/cpp. We define an arbitrarily large
// array (of size max_prob_param) and only fill the values we need for a given
// problem. Consequently, we need to fill the rest with a default error value.
// This is it.
#define PAR_ERR_VAL -99

/////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                 //
// AmrLevel-derived class for solving the Klein-Gordon equation in an expanding
// universe.          //
//                                                                                                 //
// This class does the bulk of the work; it holds all the state data, knows how
// to time-evolve     // the state data, and knows how to read inputs and write
// outputs.                                 //
//                                                                                                 //
// This is a pure abstract class! It must be derived to be used, and the derived
// class should      // specify the exact physics to be solved. //
//                                                                                                 //
/////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseAx : public amrex::AmrLevel {
public:
  //// Mandatory overrides ////////

  static const int max_prob_param =
      10; // The number of problem parameters, obviously unique to a given
          // simulation, but this is difficult to implement, so we just define
          // it to be a number that should suffice for any simulation. Change
          // this if you need more.
  static int verbose;

  //
  // Default constructor.  Builds invalid object.
  //
  BaseAx();

  //
  // The basic constructor.
  //
  BaseAx(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
         const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
         amrex::Real time);

  //
  // Advance grids at this level in time.
  //
  virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                              int ncycle) = 0;

  //
  // Initialize grid data at problem start-up.
  //
  virtual void initData() = 0;

  // This is called by NyxBld, it instantiates the global/static variables in
  // this class and reads the settings from the input file. Calls read_params()
  // and init_vars().
  static void variable_setup();

  //
  // Define tagging functions.
  //
  static void error_setup();

  //
  // Cleanup data descriptors at end of run.
  //
  static void variable_cleanup();

  //// Optional overrides ////////

  //
  // The destructor.
  //
  virtual ~BaseAx();

  //
  // Restart from a checkpoint file.
  //
  virtual void restart(amrex::Amr &papa, std::istream &is,
                       bool b_read_special = false);

  //
  // Call amrex::AmrLevel::checkPoint and then add radiation info
  //
  virtual void checkPoint(const std::string &dir, std::ostream &os,
                          amrex::VisMF::How how, bool dump_old);

  // Override if necessary: stuff to do before and after checkpoints.
  // virtual void checkPointPre(const std::string& dir, std::ostream& os);
  // virtual void checkPointPost(const std::string& dir, std::ostream& os);

  // A string written as the first item in `write_plot_file()` at level zero.
  // It is so we can distinguish between different types of plot files. For
  // BaseAx it has the form: Nyx-Vxxx.
  virtual std::string thePlotFileType() const;

  // Override this function if you would like to add plot variables to output
  // files. Nyx uses this to add some "rank" and "X(H)" and "X(He)" variables,
  // which I don't know. virtual void setPlotVariables() = {
  // AmrLevel::setPlotVariables();  }

  //
  // Write a plotfile to specified directory.
  //
  // Override if necessary: A function to tell the program how to output
  // plotfiles. Currently just calls AmrLevel::writePlotFile(...) which does
  // what we need.
  virtual void writePlotFile(const std::string &dir, std::ostream &os,
                             amrex::VisMF::How how) {
    AmrLevel::writePlotFile(dir, os, how);
  }

  // Override if necessary: A function to do something before writing plot
  // files. E.g., for particles:
  /* virtual void writePlotFilePre(const std::string& dir, std::ostream& os)
  {
      amrex::Gpu::LaunchSafeGuard lsg(true);
      if(write_skip_prepost == 1)
      {
          amrex::Print()<<"Skip writePlotFilePre"<<std::endl;
      }
      else
      {
          if(Nyx::theDMPC()) {
              Nyx::theDMPC()->WritePlotFilePre();
          }
      }
  }; */

  // Override if necessary: A function to do something after writing plot files.
  // E.g., for particles:
  virtual void writePlotFilePost(const std::string &dir, std::ostream &os);

  /////////////////////////////////////////////////////
  /////////////////////////////////////////////////////

  // Write a job info file into the plotfile/chkfile directories if desired.
  // (Not from Amrex).
  virtual void writeJobInfo(const std::string &dir);

  //
  // This amrex::MultiFab is used for the level coarser than this level to mask
  // out this level.  We only build this when it is needed. This amrex::MultiFab
  // has to live on this level even though it is at the resolution of the next
  // coarser level because it must be updated whenever this level changes.
  //
  amrex::MultiFab *fine_mask;
  amrex::MultiFab *build_fine_mask();

protected:
  /////////////////////////////////////////////////////
  ///////  Variables  /////////////////////////////////
  /////////////////////////////////////////////////////

  // N.B.
  // int level; is an important property inherited from AmrLevel, indicates the
  // AMR level associated with this object.  --PH Amr * parent; similarly, this
  // is an important property inherited from AmrLevel, pointer to the Amr object
  // that owns this AmrLevel.  --PH

  static int State_for_Time; // This must be declared in the derived class

  static int NUM_STATE; // This must be declared in the derived class

  ///// Figure out if I actually need these! --PH /////
  // static int NumSpec;
  static int strict_subcycling;
  // static amrex::Real tagging_base;
  //
  // Write all parameters into specified directory.
  //
  static int write_parameters_in_plotfile;
  static int write_grid_file;
  static int write_skip_prepost;

  static int runlog_precision;

  //
  // Static data members.
  //
  static bool dump_old;

  static amrex::ErrorList err_list;
  static amrex::Vector<amrex::AMRErrorTag> errtags;
  static amrex::BCRec phys_bc;

  static int nsteps_from_plotfile;
  static amrex::IntVect Nrep; // how many times the initial conditions are
                              // replicated in each direction

  bool FillPatchedOldState_ok;

  // specifies the memory priority
  static int minimize_memory;

  // for keeping track of the amount of CPU time used -- this will persist
  // after restarts
  static amrex::Real previousCPUTimeUsed;
  static amrex::Real startCPUTime;

  static amrex::Real fixed_dt;
  static amrex::Real initial_dt;
  static amrex::Real dt_cutoff;

  /////////////////////////////////////////////////////

  /////////////////////////////////////////////////////
  ///////  Functions  /////////////////////////////////
  /////////////////////////////////////////////////////

  ////// Mandatory overrides ////////////////////

  // Initial Conditions
  virtual void
  prob_initdata_pos(const int i, const int j, const int k,
                    amrex::Array4<amrex::Real> const &fields,
                    amrex::GeometryData const &geomdata,
                    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param>
                        &prob_param) = 0;
  // #ifdef BL_USE_MPI
  // initial condition at i,j,k in momentum-space. "state" contains field data
  // (to be initialized here), "geomdata" contains the problem geometry, and
  // "prob_param" contains the relevant problem parameters.
  virtual void
  prob_initdata_mom(const int i, const int j, const int k,
                    // amrex::Array4<amrex::Real> const& fields,
                    amrex::Array4<amrex::GpuComplex<amrex::Real>> const &fields,
                    amrex::GeometryData const &geomdata,
                    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param>
                        &prob_param) = 0;
  // #endif // BL_USE_MPI

  // This is where we tell the program how to spit out log data
  virtual void write_info() = 0;

  // Reads the input file.   N.B.: Program will compile if this is not
  // overridden,
  //                               but you *will* be missing things without it.
  static void read_params();

  // Sets up the basic run data
  static void init_vars();

  //
  // Estimate time step.
  //
  // This depends on the physics, so must be implemented in the derived class.
  // --PH
  virtual amrex::Real est_time_step(amrex::Real dt_old) = 0;

  // Derived classes must know what parameters are needed for filling in the
  // initial conditions. But this also must be static, so just remember to
  // implement it! static void
  // prob_param_fill(amrex::GpuArray<amrex::Real,max_prob_param>& prob_param) =
  // 0;

  ////// Optional overrides ////////////////////

  virtual void write_parameter_file(const std::string &dir);
  //
  // Initialize grid data from a plotfile at problem start-up.
  //
  void init_from_plotfile();

  //
  // Initialize grid data from a plotfile at problem start-up.
  //
  void ReadPlotFile(bool first, const std::string &plot_file_name,
                    bool &rhoe_infile);

  //
  // Set time levels of state data.
  //
  virtual void setTimeLevel(amrex::Real time, amrex::Real dt_old,
                            amrex::Real dt_new);

  //
  // Initialize data on this level from another BaseAx (during regrid).
  //
  virtual void init(amrex::AmrLevel &old);

  //
  // Initialize data on this level after regridding if old level did not
  // previously exist
  //
  virtual void init();

  //
  // Proceed with next timestep?
  //
  virtual int okToContinue();

  //
  // Tell amrex::Amr to write a plotfile now
  //
  // bool writePlotNow();

  //
  // Compute initial time step.
  //
  amrex::Real initial_time_step();

  //
  // Compute initial `dt'.
  //
  virtual void computeInitialDt(int finest_level, int sub_cycle,
                                amrex::Vector<int> &n_cycle,
                                const amrex::Vector<amrex::IntVect> &ref_ratio,
                                amrex::Vector<amrex::Real> &dt_level,
                                amrex::Real stop_time);

  //
  // Compute new `dt'.
  //
  virtual void computeNewDt(int finest_level, int sub_cycle,
                            amrex::Vector<int> &n_cycle,
                            const amrex::Vector<amrex::IntVect> &ref_ratio,
                            amrex::Vector<amrex::Real> &dt_min,
                            amrex::Vector<amrex::Real> &dt_level,
                            amrex::Real stop_time, int post_regrid_flag);

  //
  // Do work after timestep().
  //
  virtual void post_timestep(int iteration);

  //
  // Contains operations to be done only after a full coarse timestep.
  //
  virtual void postCoarseTimeStep(amrex::Real cumtime);

  //
  // Do work after `regrid()`.
  //
  virtual void post_regrid(int lbase, int new_finest);

  //
  // Do work after a `restart()`.
  //
  virtual void post_restart();

  //
  // Do work after `init()`.
  //
  virtual void post_init(amrex::Real stop_time);

  //
  // Error estimation for regridding.
  //
  virtual void errorEst(amrex::TagBoxArray &tb, int clearval, int tagval,
                        amrex::Real time, int n_error_buf = 0, int ngrow = 0);

  //
  // Error estimation for regridding.
  //
  virtual int WorkEstType() { return 0; }

  //
  // Called in grid_places after other tagging routines to modify
  //   the list of tagged points
  //
  virtual void
  manual_tags_placement(amrex::TagBoxArray &tags,
                        const amrex::Vector<amrex::IntVect> &bf_lev) override;

  // Returns a amrex::MultiFab containing the derived data for this level. The
  // user is responsible for deleting this pointer when done with it. If `ngrow`
  // > 0 the amrex::MultiFab is built on the appropriately grown
  // amrex::BoxArray.
  virtual std::unique_ptr<amrex::MultiFab> derive(const std::string &name,
                                                  amrex::Real time, int ngrow);

  // This version of `derive()` fills the dcomp'th component of mf with the
  // derived quantity.
  virtual void derive(const std::string &name, amrex::Real time,
                      amrex::MultiFab &mf, int dcomp);

  static void set_simd_width(const int simd_width);
  static void alloc_simd_vec();
  static void dealloc_simd_vec();

  //! Get the level directory names
  void LevelDirectoryNames(const std::string &dir, const std::string &secondDir,
                           std::string &LevelDir, std::string &FullPath);

  //! Create the Level_ and other directories in checkpoint and plot files
  virtual void CreateLevelDirectory(const std::string &dir);

  // Something about restarting from plot files.
  virtual void average_down() = 0;
  virtual void average_down(int state_indx);

  inline BaseAx &get_level(int my_level) {
    return *(BaseAx *)&parent->getLevel(my_level);
  }

  // This is only used for particles, so should be in a particle-specific class.
  /* void BaseAx::writeMultiFabAsPlotFile(const std::string& pltfile,
                       const amrex::MultiFab&    mf,
                       std::string        componentName) */

  static amrex::Real getCPUTime();

  // Implement this separately in Exec folders if you want. Otherwise, just
  // specify refinement indicators in your inputs file. Makes more sense that
  // way anyway. static void
  // prob_errtags_default(amrex::Vector<amrex::AMRErrorTag>& errtags)
  // {
  // //    AMRErrorTagInfo info;
  // //
  // errtags.push_back(AMRErrorTag(1,AMRErrorTag::GREATER,"KG_density",info));
  // };

  void print_grids(
      const std::string
          &file); // Print out the grids in "initial_grid_file" format (a unique
                  // format that doesn't have a built-in print method in AMReX).

  // Initial Conditions. No need to override these.
  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void prob_initdata_pos_on_box(
      const amrex::Box &bx, amrex::Array4<amrex::Real> const &fields,
      amrex::GeometryData const &geomdata,
      const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {
    // This just loops over all the elements in a box of lattice points.
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      prob_initdata_pos(i, j, k, fields, geomdata, prob_param);
    });
  }

  AMREX_GPU_DEVICE AMREX_FORCE_INLINE void prob_initdata_mom_on_mf(
      amrex::MultiFab &mf, amrex::Geometry const &geom,
      const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {
    BL_PROFILE("prob_initdata_mom_on_mf")

    amrex::FFT::R2C fft(geom.Domain());
    auto const &[ba, dm] = fft.getSpectralDataLayout();

    amrex::cMultiFab fillK(ba, dm, mf.nComp(), 0);

    // Initialize the Fourier data
    for (amrex::MFIter mfi(fillK, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const amrex::Box &bx = mfi.tilebox();
      const auto fab_new = fillK.array(mfi);

      amrex::ParallelFor(
          bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
          // NOTE THE & instead of =!! This takes references for external
          // variables as opposed to making local copies.
          {
            prob_initdata_mom(i, j, k, fab_new, geom.data(), prob_param);
	    
            // Set the 0-mode to 0 since we add it in later
            if (i == 0 && j == 0 && k == 0) {
              for (int comp = 0; comp < fillK.nComp(); comp++)
                fab_new(0, 0, 0, comp) = amrex::GpuComplex(
                    0., 0.); // Set the 0-mode to 0 since we add it in later
            }
          });
    }

    // amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::backward>
    // fft_backward(geom.Domain()); fft_backward.backward(fillK, mf);
    fft.backward(fillK, mf);
    mf.mult(1. / mf.boxArray().numPts());
    mf.FillBoundary(geom.periodicity());
  }

}; // BaseAx Class

#endif // BASE_AX
