#ifndef _Newtonian_H_
#define _Newtonian_H_

// Let's see if we can do this all in just the two files.

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
          int _field_comp, int field_comp, int pstate_comp, int gstate_comp);

  // Destructor: Cleans up the Gravity object
  virtual ~Gravity();
  
  // In IO
  static void read_params(); // Reads parameters for the gravitational solver
  void write_info();
  void writePlotFilePost(const std::string &dir, std::ostream &os);
  void outputPowerSpectrum(const std::string &dir);
};
#endif
