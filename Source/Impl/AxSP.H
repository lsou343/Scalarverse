#ifndef _AX_SP_H_
#define _AX_SP_H_
// #include <AxSCHComov.H>
#include <BaseGrav.H>
#include <Comoving_EOS.H>
#include <Gravity.H>
//  #include <BaseAx.H> // Or <AxSCHComov.H> if you prefer direct inheritance
//  from AxSCHComov

/*
/////////////////////////////////////////////////////////////////////
// Implementation of BaseAx (or AxSCHComov) to solve the Schrödinger-Poisson //
// equation in a consistently-evolved comoving background.         //
//                                                                  //
/////////////////////////////////////////////////////////////////////
*/

class AxSP : public BaseGrav {
  // Inherit from AxSCHComov, which inherits from BaseAx
  // Inherit from BaseGrav to include gravity fields {
public:
  ////////////////////////////////////////////////////////////
  /////////    Constructors / Destructor     ////////////////
  ////////////////////////////////////////////////////////////

  // Default constructor
  AxSP();
  // Full constructor
  AxSP(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
       const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
       amrex::Real time);

  virtual ~AxSP();

  ////////////////////////////////////////////////////////////
  /////////    Mandatory Functions           /////////////////
  ////////////////////////////////////////////////////////////

  // Initialize data on this level (and possibly do comoving-specific setup).
  virtual void initData() override;

  // Must define a "variable_setup()" that sets up descriptors
  // or any additional states needed for comoving runs.
  static void variable_setup();

  ////////////////////////////////////////////////////////////
  /////////    IO / Checkpointing Functions   ///////////////
  ////////////////////////////////////////////////////////////

  // // Write out any info relevant to comoving runs
  // virtual void write_info() override;
  //
  // // Called after a plotfile is written
  // virtual void writePlotFilePost(const std::string &dir,
  //                                std::ostream &os) override;
  //
  // // If you want to store scale factor, a(t), etc. in checkpoint
  // virtual void checkPointPost(const std::string &dir,
  //                             std::ostream &os) override;

  // Add gravity fields in a new AxSP::Fields
  enum class Fields : unsigned char {
    // Start from the last SCH field
    // Dens = 0, // Mean Density
    // SCHf_Re,  // Real part
    // SCHf_Im,  // Imaginary part
    // Phase,    // Phase
    PhiGrav = 4, // static_cast<unsigned char>(AxSCH::Fields::Phase) + 1,
    GradPhi_X,
    GradPhi_Y,
    GradPhi_Z
  };

  enum class StateType : unsigned char {
    // SCH_Type = 0,
    // // static_cast<unsigned char>(AxSCH::StateType::SCH_Type) + 1,
    PhiGrav_Type = 1,
    Gravity_Type,
  };
  // For convenience, forward the same enumerators / indexing:
  // Fields:
  static int nFields(); // 4 gravity fields
  static int getField(Fields f);
  static Fields getField(int f);
  static int nStates();
  static int getState(StateType st);
  static StateType getState(int st);

  // Retrieved general density field from Base code
  MultiFab &get_density(bool old = false) override;

protected:
  ////////////////////////////////////////////////////////////
  /////////    Overridden from the base class   /////////////
  ////////////////////////////////////////////////////////////
  // int BaseAx::NUM_STATE = nStates(); // Define the number of states
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
  // static void read_params();

  virtual void write_info() override;

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

#endif // _AX_SP_H_
