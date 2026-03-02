#ifndef _Gravity_H_ // Include guard to prevent multiple inclusions of the same
                    // header file
#define _Gravity_H_

#include <AMReX_AmrLevel.H>     // Include AMReX AmrLevel class header
#include <AMReX_FluxRegister.H> // Include AMReX FluxRegister class header
#ifdef AMREX_PARTICLES
#include <AMReX_Particles.H> // Include AMReX Particles class header if AMREX_PARTICLES is defined
#endif
#include <AMReX_MLCellLinOp.H>
#include <AMReX_MLLinOp.H>   // Include AMReX MLLinOp class header
#include <AMReX_MLPoisson.H> // Include AMReX MLPoisson class header

// ************************************************************************** //

// Ggravity is defined as -4 * pi * G, where G is the gravitational constant.
// G is defined as Gconst in `/Source/Constants/constants_cosmo.H` if

// In CGS, this constant is currently
//      Gconst   =  6.67428e-8           cm^3/g/s^2 , which results in
//      Ggravity = -83.8503442814844e-8  cm^3/g/s^2

// In cosmological units, this constant is currently
//      Gconst   =   4.3   e-9  (km/s)^2 Mpc / Msun, which results in
//      Ggravity = -54.0354e-9  (km/s)^2 Mpc / Msun

// ************************************************************************** //

extern amrex::Real Ggravity;

// Definition of the Gravity class
class Gravity {

public:
  static amrex::Real Gconst;

  // Constructor: Initializes a Gravity object
  Gravity(amrex::Amr *Parent, int _finest_level, amrex::BCRec *_phys_bc,
          int _Density_comp, int state_comp, int pstate_comp, int gstate_comp);

  // Destructor: Cleans up the Gravity object
  virtual ~Gravity();

  // In IO
  static void read_params(); // Reads parameters for the gravitational solver
  void write_info();
  void writePlotFilePost(const std::string &dir, std::ostream &os);
  void outputPowerSpectrum(const std::string &dir);

  // Installs data for a specific level
  void install_level(int level, amrex::AmrLevel *level_data_to_install);

  // Returns the no_sync flag
  int get_no_sync();

  // Returns the no_composite flag
  int get_no_composite();

  // Sets the mass offset for a specific time
  void set_mass_offset(amrex::Real time);

  // Returns the gradient of phi at the previous time step for a given level
  amrex::Vector<amrex::MultiFab *> get_grad_phi_prev(int level);

  // Returns the gradient of phi at the current time step for a given level
  amrex::Vector<amrex::MultiFab *> get_grad_phi_curr(int level);

  // Adds a MultiFab to the current phi at a specific level
  void plus_phi_curr(int level, amrex::MultiFab &addend);

  // Adds a vector of MultiFabs to the current gradient phi at a specific level
  void plus_grad_phi_curr(int level,
                          const amrex::Vector<amrex::MultiFab *> &addend);

  //
  // Moves curr phi data to prev and resets curr to 1.e50
  //

  // Swaps the current and previous time levels for phi data
  void swap_time_levels(int level);

  // Sets up the Poisson solver between coarse and fine levels
  void setup_Poisson(int crse_level, int fine_level);

  // Solves the Poisson equation using the MLMG solver
  amrex::Real solve_with_MLMG(
      int crse_level, int fine_level,
      const amrex::Vector<amrex::MultiFab *> &phi,
      const amrex::Vector<const amrex::MultiFab *> &rhs,
      const amrex::Vector<std::array<amrex::MultiFab *, AMREX_SPACEDIM>>
          &grad_phi,
      const amrex::MultiFab *const crse_bcdata, amrex::Real rel_eps,
      amrex::Real abs_eps);

  // Sets boundary conditions for the Poisson solver
  void set_boundary(amrex::BndryData &bd, amrex::MultiFab &rhs,
                    const amrex::Real *dx);

  // Solves for the old gravitational potential phi
  void solve_for_old_phi(int level, amrex::MultiFab &phi,
                         const amrex::Vector<amrex::MultiFab *> &grad_phi,
                         int fill_interior, int grav_n_grow = 1);

  // Solves for the new gravitational potential phi
  void solve_for_new_phi(int level, amrex::MultiFab &phi,
                         const amrex::Vector<amrex::MultiFab *> &grad_phi,
                         int fill_interior, int grav_n_grow = 1);

  // Solves for phi at a specific time
  void solve_for_phi(int level, amrex::MultiFab &Rhs, amrex::MultiFab &phi,
                     const amrex::Vector<amrex::MultiFab *> &grad_phi,
                     amrex::Real time, int fill_interior);

  // Solves for the change in phi
  void solve_for_delta_phi(
      int crse_level, int fine_level, amrex::MultiFab &CrseRhs,
      const amrex::Vector<amrex::MultiFab *> &delta_phi,
      const amrex::Vector<amrex::Vector<amrex::MultiFab *>> &grad_delta_phi);

  // Synchronizes gravity across levels
  void gravity_sync(int crse_level, int fine_level, int iteration, int ncycle,
                    const amrex::MultiFab &drho_and_drhoU,
                    const amrex::MultiFab &dphi,
                    const amrex::Vector<amrex::MultiFab *> &grad_delta_phi_cc);

  // Multilevel solver for old phi
  void multilevel_solve_for_old_phi(int level, int finest_level,
                                    int ngrow_for_solve,
                                    int use_previous_phi_as_guess = 0);

  // Multilevel solver for new phi
  void multilevel_solve_for_new_phi(int level, int finest_level,
                                    int ngrow_for_solve,
                                    int use_previous_phi_as_guess = 0);

  // Performs the actual multilevel solve
  void actual_multilevel_solve(
      int level, int finest_level,
      const amrex::Vector<amrex::Vector<amrex::MultiFab *>> &grad_phi,
      int is_new, int ngrow_for_solve, int use_previous_phi_as_guess = 0);

  // Retrieves the coarse level gradient phi
  void get_crse_grad_phi(
      int level, amrex::Vector<std::unique_ptr<amrex::MultiFab>> &grad_phi_crse,
      amrex::Real time);

  // Retrieves the coarse level phi
  void get_crse_phi(int level, amrex::MultiFab &phi_crse, amrex::Real time);

  // Resets the phi flux registers to zero
  void zero_phi_flux_reg(int level);

  // Retrieves the old gravitational vector
  void get_old_grav_vector(int level, amrex::MultiFab &grav_vector,
                           amrex::Real time);

  // Retrieves the new gravitational vector
  void get_new_grav_vector(int level, amrex::MultiFab &grav_vector,
                           amrex::Real time);

  // Averages fine edge-centered data onto coarse edge-centered data
  void average_fine_ec_onto_crse_ec(int level, int is_new);

  // Adds data to fluxes
  void add_to_fluxes(int level, int iteration, int ncycle);

  // Refluxes phi data between levels
  void reflux_phi(int level, amrex::MultiFab &dphi);

  // Creates multigrid boundary conditions
  void make_mg_bc();

  // Arrays of boundary condition types for MLMG solver
  std::array<amrex::MLLinOp::BCType, AMREX_SPACEDIM> mlmg_lobc;
  std::array<amrex::MLLinOp::BCType, AMREX_SPACEDIM> mlmg_hibc;

  // Sets Dirichlet boundary conditions
  void set_dirichlet_bcs(int level, amrex::MultiFab *phi);

protected:
  // Pointer to the AMR object
  amrex::Amr *parent;

  // Vector of pointers to AMR levels
  amrex::Vector<amrex::AmrLevel *> LevelData;

  // Vectors of unique pointers to MultiFabs storing gradient of phi at current
  // and previous time steps
  amrex::Vector<amrex::Vector<std::unique_ptr<amrex::MultiFab>>> grad_phi_curr;
  amrex::Vector<amrex::Vector<std::unique_ptr<amrex::MultiFab>>> grad_phi_prev;

  // Vector of unique pointers to flux registers for phi
  amrex::Vector<std::unique_ptr<amrex::FluxRegister>> phi_flux_reg;

  // References to vectors of BoxArray and DistributionMapping defining grid
  // structure and distribution mapping
  const amrex::Vector<amrex::BoxArray> &grids;
  const amrex::Vector<amrex::DistributionMapping> &dmap;

  // Vector of reals storing residual norm of the solver at each level
  amrex::Vector<amrex::Real> level_solver_resnorm;

  // Integer representing the density component
  int density;
  int State_Type; // Assuming these are the correct values
  int PhiGrav_Type;
  int Gravity_Type;

  // Integer representing the finest level currently allocated
  int finest_level_allocated;

  // Pointer to boundary condition records
  amrex::BCRec *phys_bc;

  // Unique pointer to MLPoisson object for solving Poisson's equation
  std::unique_ptr<amrex::MLPoisson> mlpoisson;

  // Static variables for various settings and tolerances
  static int verbose;
  static int mg_verbose;
  static std::string mg_bottom_solver;
  static int no_sync;
  static int no_composite;
  static int dirichlet_bcs;
  static int mg_max_fmg_iter;
  static int mlmg_agglomeration;
  static int mlmg_consolidation;
  static amrex::Real mass_offset;
  static amrex::Real sl_tol;
  static amrex::Real ml_tol;
  static amrex::Real delta_tol;

  // Adds particles to the right-hand side
  void AddParticlesToRhs(int level, amrex::MultiFab &Rhs, int ngrow);

  // Adds ghost particles to the right-hand side
  void AddGhostParticlesToRhs(int level, amrex::MultiFab &Rhs);

  // Adds virtual particles to the right-hand side
  void AddVirtualParticlesToRhs(int level, amrex::MultiFab &Rhs, int ngrow);

  // Adds particles to the right-hand side across multiple levels
  void AddParticlesToRhs(int base_level, int finest_level, int ngrow,
                         const amrex::Vector<amrex::MultiFab *> &Rhs_particles);

  // Adds ghost particles to the right-hand side across multiple levels
  void
  AddGhostParticlesToRhs(int level,
                         const amrex::Vector<amrex::MultiFab *> &Rhs_particles);

  // Adds virtual particles to the right-hand side across multiple levels
  void AddVirtualParticlesToRhs(
      int finest_level, const amrex::Vector<amrex::MultiFab *> &Rhs_particles);

  // Corrects the right-hand side using the mass offset
  void CorrectRhsUsingOffset(int level, amrex::MultiFab &Rhs);
};
#endif // _Gravity_H_
