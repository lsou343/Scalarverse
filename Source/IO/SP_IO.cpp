#include <AxSP.H>
#include <Comoving_EOS.H>

#include <ostream>
#include <unistd.h>

// // Read in the parameters specific to a SCH run
// void AxSCHComov::read_params() { AxSCH::read_params(); }
using namespace amrex;
void AxSP::write_info() {
  int ndatalogs = parent->NumDataLogs();
  Real time_unit =
      1.0; // 3.0856776e19 / 31557600.0; // conversion to Julian years
  //
  // Print() << "AxSCH::write_info() level:" << level << std::endl;

  int rlp = AxSP::runlog_precision;

  if (ndatalogs > 0) {

    // Real time =
    //     state[AxSCH::getState(AxSCH::StateType::SCH_Type)].curTime();

    Real time = state[AxSP::getState(AxSP::StateType::PhiGrav_Type)].curTime();

    Real dt = parent->dtLevel(0);
    int nstep = parent->levelSteps(0);

    //// N.B. This needs to be updated for refined data
    //// Print the average value of the field and its derivative
    int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                   geom.Domain().length(2); // Assuming a rectangular domain

    // std::unique_ptr<MultiFab> Psi_Re = AxSCH::derive("SCHf_Re", time,
    // 0); std::unique_ptr<MultiFab> Psi_Im = AxSCH::derive("SCHf_Im",
    // time, 0);
    std::unique_ptr<MultiFab> rho = AxSCH::derive("Dens", time, 0);
    std::unique_ptr<MultiFab> phi_pot = AxSP::derive("phi_grav", time, 0);

    // Real avPsi_Re = Psi_Re->sum() / gridsize;
    // Real avPsi_Im = Psi_Im->sum() / gridsize;
    Real avrho = rho->sum() / gridsize;
    Real maxrho = rho->max(0);

    // highest overdensity
    Real maxoverdens = (maxrho / avrho) - 1.0;

    Real avphi = phi_pot->sum() / gridsize;
    Real maxphi = phi_pot->max(0);

    // Real maxphi =
    // get_new_data(getState(AxSP::StateType::PhiGrav_Type)).max(0);
    // scale factor
    Real a = Comoving::get_comoving_a(time);

    // jeans length kJ = (16*pi*G*a*avg_rho_comov/hbaroverm^2)^(1/4)
    // if (time == 0.0) {
    // Real avrho_init = rho->sum() / gridsize;
    //   Real a_init = Comoving::get_comoving_a(time);
    //   Real avrho_end = avrho_init * a_init * a_init * a_init;
    // }
    Real avrho_comov =
        avrho * a * a * a; // average density in comoving coordinates
    Real kJ = std::pow((16 * M_PI * Gravity::Gconst * a * avrho_comov /
                        AxSCH::hbaroverm / AxSCH::hbaroverm),
                       1 / 4.);

    // dispersion relation c_s_squered = hbaroverm**2 * k**2 / 4.0 / a**2
    // omega_squered = (c_s_squered * k**2 / a**2 - 4 * np.pi * G * meandens)
    // Real L_box_comov = geom.ProbHi(0) - geom.ProbLo(0);
    // Real k_wavenumber =
    // AxSCH::SCH_k0 * 2.0 * M_PI /
    // L_box_comov; // convert mode number to // Needs to be 2pi
    // Real c_s_squered = (AxSCH::hbaroverm * AxSCH::hbaroverm) * k_wavenumber *
    // k_wavenumber / 4.0 / a / a;
    // Real omega_squared = c_s_squered * k_wavenumber * k_wavenumber / a / a -
    //                      4.0 * M_PI * Gravity::Gconst * avrho_comov;

    if (ParallelDescriptor::IOProcessor()) {
      std::ostream &data_loga = parent->DataLog(0);
      if (time == 0.0) {
        data_loga << std::setw(8) << "#     nstep";
        data_loga << std::setw(14) << "   time    ";
        data_loga << std::setw(14) << "     dt      ";
        // data_loga << std::setw(14) << "  <Psi_Re>     ";
        // data_loga << std::setw(14) << " <Psi_Im>     ";
        data_loga << std::setw(14) << "     <rho>     ";
        data_loga << std::setw(14) << "   rhomax     ";
        data_loga << std::setw(14) << "overdensmax";
        // data_loga << std::setw(14) << "     <phi>      ";
        data_loga << std::setw(14) << "     phimax     ";
        data_loga << std::setw(14) << "   efolds    ";
        data_loga << std::setw(14) << " a    ";
        data_loga << std::setw(14) << "kJ  ";
        // data_loga << std::setw(14) << "omega^2 ";
        data_loga << std::endl;
      }
      data_loga << std::setw(8) << nstep;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << (time + Comoving::initial_time) * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp) << dt * time_unit;
      // data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Re;
      // data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Im;
      data_loga << std::setw(14) << std::setprecision(rlp) << avrho;
      data_loga << std::setw(14) << std::setprecision(rlp) << maxrho;
      data_loga << std::setw(14) << std::setprecision(rlp) << maxoverdens;
      // data_loga << std::setw(14) << std::setprecision(rlp) << avphi;
      data_loga << std::setw(14) << std::setprecision(rlp) << maxphi;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << std::log(a / Comoving::initial_a);
      data_loga << std::setw(14) << std::setprecision(rlp) << a;
      data_loga << std::setw(14) << std::setprecision(rlp) << kJ;
      // data_loga << std::setw(14) << std::setprecision(rlp) << omega_squared;
      data_loga << std::endl;
    }
  }
}

//
// void AxSCHComov::checkPointPost(const std::string &dir, std::ostream &os) {
//
//   // Write comoving_a into its own file in the checkpoint directory
//   if (ParallelDescriptor::IOProcessor()) {
//     Real time = state[getState(StateType::SCH_Type)].curTime();
//     Real dt = parent->dtLevel(0);
//     std::string FileName = dir + "/comoving_a";
//     std::ofstream File;
//     File.open(FileName.c_str(), std::ios::out | std::ios::trunc);
//     if (!File.good()) {
//       FileOpenFailed(FileName);
//     }
//     File.precision(15);
//     File << Comoving::get_comoving_a(time) << '\n';
//     // File << Comoving::get_comoving_a(time - dt) << '\n';
//     File << time << '\n';
//     // File << time - dt << '\n';
//     File.close();
//   }
// }
