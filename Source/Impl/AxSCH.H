#ifndef _AX_SCH_H_
#define _AX_SCH_H_

#include <BaseAx.H>
//////////////////////////////////////////////////////////////////////
//    Implementation of BaseAx to solve the Schroedinger-Poisson    //
//    equation. Adapted from FDM for axion-like behavior.           //
//////////////////////////////////////////////////////////////////////

class AxSCH : public BaseAx {
public:
  ////////////////////////////////////////////////////////////
  /////////    Variables       ///////////////////////////////
  ////////////////////////////////////////////////////////////
  static amrex::Real mtt;           // Mass of the field
  static amrex::Real hbaroverm;     // Reduced Planck constant over mass
  static amrex::Real vonNeumann_dt; // Von Neumann stability factor
  static amrex::Real SCH0;          // Need the initial value of the field.
  static amrex::Real SCH_k0;        // Initial fixed k mode
  static amrex::Real Phase0;        // Initial phase of the field.
  static amrex::Real sigma;         // Width of the Gaussian
  // levelmethod for the root grif: 0 = FD, 1 = PS
  static int PSorFD;
  static int PSorder;
  static amrex::Real test_factor; // Factor for testing, that way one does not
                                  // need to comile every time
  // If initialized from KG field
  static std::string KGinitDirName; // Directory name for MultiFab
  static amrex::Real KGm;           // Mass of the field
  static amrex::Real KGA;           // A parameter
  static amrex::Real KGB;           // B parameter
  static amrex::Real KGr;           // r parameter
  static amrex::Real KGs;           // s parameter
  //
  ////////////////////////////////////////////////////////////
  /////////    Functions       ///////////////////////////////
  ////////////////////////////////////////////////////////////

  ///////////////
  // Mandatory //
  ///////////////

  // In Impl
  AxSCH();
  AxSCH(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
        const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
        amrex::Real time);
  virtual ~AxSCH() {};
  virtual void initData() override;
  static void variable_setup();

  // In Timestep
  virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                              int ncycle) override;

  // In IO
  virtual void write_info() override;
  virtual void writePlotFilePost(const std::string &dir,
                                 std::ostream &os) override;
  // virtual void checkPointPost (const std::string& dir, std::ostream& os);
  // virtual void writeJobInfo(const std::string& dir);

  ////////////////////////////////////////////////////////////
  /////////    Types           ///////////////////////////////
  ////////////////////////////////////////////////////////////

  ////////////////////
  // Optional/Local //
  ////////////////////
  // void fill_rho();
  virtual void read_KG_MultiFab(std::string mfDirName, int nghost,
                                amrex::MultiFab &mf);

  //
  // Error estimation for regridding.
  //
  virtual void errorEst(amrex::TagBoxArray &tb, int clearval, int tagval,
                        amrex::Real time, int n_error_buf = 0,
                        int ngrow = 0) override;

  // Finally for public, keep track of the different types of particles/fields
  // involved, and the simulation type. Must be specified by the derived class
  // (it will only matter to them anyway).

  enum class StateType : unsigned char {
    SCH_Type = 0,
  };

  enum class Fields : unsigned char {
    Dens = 0, // Mean Density
    SCHf_Re,  // Real part
    SCHf_Im,  // Imaginary part
    Phase,    // Phase
  };

  enum class ICType : unsigned char {
    test = 0,
    plain_wave,
    gaussian,
    KGfield,
    linearperurbation,
  };

  static int nFields();
  static int getField(Fields f);
  static Fields getField(int f);
  static int nStates();
  static int getState(StateType st);
  static StateType getState(int st);
  static int getIC(ICType it);
  static ICType getIC(int it);

protected:
  ////////////////////////////////////////////////////////////
  /////////    Variables       ///////////////////////////////
  ////////////////////////////////////////////////////////////
  // ALready stated in BaseAx.H
  // int BaseAx::NUM_STATE = nStates(); // Define the number of states
  // int BaseAx::State_for_Time =
  //     0; // Define the state for time variable. Must ensure the state used to
  //        // define the time variable is mapped to 0 in getState().
  //
  static ICType ic; // Which IC to use

  ////////////////////////////////////////////////////////////
  /////////    Functions       ///////////////////////////////
  ////////////////////////////////////////////////////////////

  // In Impl
  virtual void init(amrex::AmrLevel &old) override;
  virtual void init() override;

  // This depends on the physics, so must be implemented in the derived class.
  // --PH
  amrex::Real est_time_step(amrex::Real dt_old) override;
  virtual void average_down() override;

  // In IO
  static void read_params();
  static void prob_param_fill(
      amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_params);

  // In Prob.cpp
  virtual void
  prob_initdata_pos(const int i, const int j, const int k,
                    amrex::Array4<amrex::Real> const &fields,
                    amrex::GeometryData const &geomdata,
                    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param>
                        &prob_param) override;
#ifdef BL_USE_MPI
  // initial condition at i,j,k in momentum-space. "state" contains field data
  // (to be initialized here), "geomdata" contains the problem geometry, and
  // "prob_param" contains the relevant problem parameters.
  virtual void
  prob_initdata_mom(const int i, const int j, const int k,
                    amrex::Array4<amrex::GpuComplex<amrex::Real>> const
                        &fields, // Change Real -> GpuComplex<Real>
                    amrex::GeometryData const &geomdata,
                    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param>
                        &prob_param) override;

  void prob_initdata_mom_on_mf(
      amrex::MultiFab &mf, amrex::Geometry const &geom,
      const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param);
#endif // BL_USE_MPI

  ////////////////////
  // Optional/Local //
  ////////////////////

  // In Impl
  // void fill_rho(amrex::MultiFab &mf, amrex::Real invdelsq);

  // In Timestep

  // In IO
  void outputPowerSpectrum(const std::string &dir);
};
#endif // AX_SCH_H
