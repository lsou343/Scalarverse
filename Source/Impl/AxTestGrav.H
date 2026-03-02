#ifndef AX_TEST_GRAV_H
#define AX_TEST_GRAV_H

#include <BaseAx.H> // Include the header for BaseAx to resolve the base class
#include <BaseGrav.H>
#include <Gravity.H>

// AMReX
#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <iostream>

using namespace amrex;

class AxTestGrav
    :
    public BaseGrav
{
public:
  AxTestGrav();
  AxTestGrav(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
             const amrex::BoxArray &ba, const amrex::DistributionMapping &dm,
             amrex::Real time);

  virtual ~AxTestGrav() override;

  // Retrieved general density field from Base code
  MultiFab &get_density(bool old = false) override;

  // Enums for managing fields
  enum class Fields : unsigned char {
    Density = 0, // Density field
    PhiGrav,     // Gravitational potential
    GradPhi_X,   // Gradient of the gravitational potential in X
    GradPhi_Y,   // Gradient of the gravitational potential in Y
    GradPhi_Z    // Gradient of the gravitational potential in Z
  };

  enum class StateType : unsigned char {
    State_Type = 0,
    PhiGrav_Type,
    Gravity_Type,
    NUM_STATE_TYPE
  };

  static int nFields();
  static int getField(Fields f);
  static int nStates();
  static int getState(StateType st);

  // In Timestep
  virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                              int ncycle) override;

  // Initialization of the density distribution
  virtual void initData() override;
  
  // In IO - LSR, taken from AxKGComov.H
  virtual void write_info ();
  virtual void writePlotFilePost(const std::string& dir, std::ostream& os);
  virtual void checkPointPost (const std::string& dir, std::ostream& os);

  // File writing and post-processing
  // virtual void write_info();
  // virtual void writePlotFilePost(const std::string &dir, std::ostream &os);

  // Variable setup for descriptors and derived fields
  static void variable_setup();

protected:
  int BaseAx::NUM_STATE = nStates(); // Define the number of states
  int BaseAx::State_for_Time =
      0; // Define the state for time variable. Must ensure the state used to
         // define the time variable is mapped to 0 in getState().

  // In Impl
  virtual void init(amrex::AmrLevel &old);
  virtual void init();

  // In IO - LSR, taken from AxKGComov.H
  // THESE NEED TO BE INCLUDED, BUT STATIC FUNCTIONS CAN'T BE FORCED IN C++, SO HEADS-UP, READ THE COMMENTS
  static void read_params(); 

  // Time-step estimation
  virtual amrex::Real est_time_step(amrex::Real dt_old) override;

  // Average down finer grid data to coarser grids
  virtual void average_down() override;

  // density initialization
  virtual void
  prob_initdata_pos(int i, int j, int k, const amrex::Array4<double> &fields,
                    const amrex::GeometryData &geomdata,
                    const amrex::GpuArray<double, 10> &prob_param) override;

  virtual void
  prob_initdata_mom(int i, int j, int k,
                    amrex::Array4<amrex::GpuComplex<amrex::Real>> const &fields,
                    const amrex::GeometryData &geomdata,
                    const amrex::GpuArray<double, 10> &prob_param) override;

//  void outputPotentialSpectrum(const std::string &dir);  // LSR - uncommented this
};

#endif // AX_TEST_GRAV_H
