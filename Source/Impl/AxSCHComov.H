#ifndef _AX_SCH_COMOV_H_
#define _AX_SCH_COMOV_H_
#include <AxSCH.H>
#include <Comoving_EOS.H>

/*
/////////////////////////////////////////////////////////////////////
// Implementation of BaseAx (or AxSCH) to solve the Schrödinger    //
// equation in a consistently-evolved comoving background.         //
//                                                                  //
// This mirrors what AxKGComov does for the Klein-Gordon code.      //
/////////////////////////////////////////////////////////////////////
*/

class AxSCHComov
    : public AxSCH // or `: public AxSCH` if you do not have a BaseSCH
{
public:
  ////////////////////////////////////////////////////////////
  /////////    Constructors / Destructor     ////////////////
  ////////////////////////////////////////////////////////////

  // Default constructor
  AxSCHComov() : AxSCH() {
    // Possibly do some comoving-specific initialization
  }

  // Full constructor
  AxSCHComov(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
             const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
             amrex::Real time)
      : AxSCH(papa, lev, level_geom, bl, dm, time) {
    // Possibly do more comoving-specific initialization
    Comoving::init_comoving_state(time, lev);
  }

  virtual ~AxSCHComov();

  ////////////////////////////////////////////////////////////
  /////////    Mandatory Functions           /////////////////
  ////////////////////////////////////////////////////////////

  // Initialize data on this level (and possibly do comoving-specific setup).
  virtual void initData() override;

  // Must define a "variable_setup()" that sets up descriptors
  // or any additional states needed for comoving runs.
  static void variable_setup();

  // Advance the solution in time for one timestep.
  // Overload or augment the base advance() to add expansion effects, etc.
  void advance_SCH_FD(amrex::Real time, amrex::Real dt, amrex::Real a_old,
                      amrex::Real a_new);
  void advance_SCH_PS(amrex::Real time, amrex::Real dt, amrex::Real a_old,
                      amrex::Real a_new);
  virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                              int ncycle) override;
  // amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
  //                     int ncycle, int stage);

  ////////////////////////////////////////////////////////////
  /////////    IO / Checkpointing Functions   ///////////////
  ////////////////////////////////////////////////////////////

  // Write out any info relevant to comoving runs
  virtual void write_info() override;

  // Called after a plotfile is written
  virtual void writePlotFilePost(const std::string &dir,
                                 std::ostream &os) override;

  // If you want to store scale factor, a(t), etc. in checkpoint
  virtual void checkPointPost(const std::string &dir,
                              std::ostream &os) override;

  ////////////////////////////////////////////////////////////
  /////////    Reuse the AxSCH Enumerations     /////////////
  ////////////////////////////////////////////////////////////

  // For convenience, forward the same enumerators / indexing:
  static int nFields() { return AxSCH::nFields(); }
  static int getField(AxSCH::Fields f) { return AxSCH::getField(f); }
  static AxSCH::Fields getField(int f) { return AxSCH::getField(f); }
  static int nStates() { return AxSCH::nStates(); }
  static int getState(AxSCH::StateType st) { return AxSCH::getState(st); }
  static AxSCH::StateType getState(int st) { return AxSCH::getState(st); }
  static int getIC(AxSCH::ICType it) { return AxSCH::getIC(it); }
  static AxSCH::ICType getIC(int it) { return AxSCH::getIC(it); }

protected:
  ////////////////////////////////////////////////////////////
  /////////    Overridden from the base class   /////////////
  ////////////////////////////////////////////////////////////

  // Called during level initialization from a coarser (old) level
  virtual void init(amrex::AmrLevel &old) override;

  // Called during level initialization when this is a new level
  virtual void init() override;

  // Typically used after a restart to read scale-factor, etc.
  // virtual void post_restart();

  // Calculate next time-step size for comoving
  virtual amrex::Real est_time_step(amrex::Real dt_old) override;

  // If you need to do any volume-averaging or downsampling
  virtual void average_down() override;

  // Comoving version of reading parameters
  static void read_params();

  // Override the virtual functions from BaseAx
  virtual void computeInitialDt(int finest_level, int sub_cycle,
                                amrex::Vector<int> &n_cycle,
                                const amrex::Vector<amrex::IntVect> &ref_ratio,
                                amrex::Vector<amrex::Real> &dt_level,
                                amrex::Real stop_time) override;

  virtual void computeNewDt(int finest_level, int sub_cycle,
                            amrex::Vector<int> &n_cycle,
                            const amrex::Vector<amrex::IntVect> &ref_ratio,
                            amrex::Vector<amrex::Real> &dt_min,
                            amrex::Vector<amrex::Real> &dt_level,
                            amrex::Real stop_time,
                            int post_regrid_flag) override;

  ////////////////////////////////////////////////////////////
  /////////    Additional Local Functions       /////////////
  ////////////////////////////////////////////////////////////

  // e.g. fill_rho() if you want to track densities in comoving frame
  // void fill_rho();

  // ... etc ...
};

#endif // _AX_SCH_COMOV_H_
