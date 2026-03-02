#include <AxSCHComov.H>
#include <Comoving_EOS.H>

#include <ostream>
#include <unistd.h>

// Read in the parameters specific to a SCH run
void AxSCHComov::read_params() { AxSCH::read_params(); }

void AxSCHComov::write_info() {
  int ndatalogs = parent->NumDataLogs();
  amrex::Real time_unit =
      1.0; // 3.0856776e19 / 31557600.0; // conversion to Julian years
  //
  // amrex::Print() << "AxSCH::write_info() level:" << level << std::endl;

  int rlp = AxSCH::runlog_precision;

  if (ndatalogs > 0) {

    amrex::Real time = state[getState(StateType::SCH_Type)].curTime();
    amrex::Real dt = parent->dtLevel(0);
    int nstep = parent->levelSteps(0);

    //// N.B. This needs to be updated for refined data
    //// Print the average value of the field and its derivative
    int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                   geom.Domain().length(2); // Assuming a rectangular domain

    // std::unique_ptr<amrex::MultiFab> Psi_Re = AxSCH::derive("SCHf_Re", time,
    // 0); std::unique_ptr<amrex::MultiFab> Psi_Im = AxSCH::derive("SCHf_Im",
    // time, 0);
    std::unique_ptr<amrex::MultiFab> rho = AxSCH::derive("Dens", time, 0);

    // amrex::Real avPsi_Re = Psi_Re->sum() / gridsize;
    // amrex::Real avPsi_Im = Psi_Im->sum() / gridsize;
    amrex::Real avrho = rho->sum() / gridsize;
    ///////

    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ostream &data_loga = parent->DataLog(0);
      if (time == 0.0) {
        data_loga << std::setw(8) << "      nstep";
        data_loga << std::setw(14) << "   time    ";
        data_loga << std::setw(14) << "     dt      ";
        // data_loga << std::setw(14) << "  <Psi_Re>     ";
        // data_loga << std::setw(14) << " <Psi_Im>     ";
        data_loga << std::setw(14) << "     <rho>     ";
        data_loga << std::setw(14) << "       a       ";
        data_loga << std::endl;
      }
      data_loga << std::setw(8) << nstep;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << (time + Comoving::initial_time) * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp) << dt * time_unit;
      // data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Re;
      // data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Im;
      data_loga << std::setw(14) << std::setprecision(rlp) << avrho;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << Comoving::get_comoving_a(time);
      data_loga << std::endl;
    }
  }
}

void AxSCHComov::writePlotFilePost(const std::string &dir, std::ostream &os) {
  AxSCH::writePlotFilePost(dir, os);
  amrex::Real time = state[getState(StateType::SCH_Type)].curTime();
  if (write_skip_prepost != 1) {
    if (verbose) {
      if (level == 0) {
        amrex::Print().SetPrecision(15)
            << "Output file " << dir << " at scale factor "
            << std::to_string(Comoving::get_comoving_a(time)) << " and step "
            << std::to_string(nStep()) << std::endl;
      }
    }
  }
}

void AxSCHComov::checkPointPost(const std::string &dir, std::ostream &os) {

  // Write comoving_a into its own file in the checkpoint directory
  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Real time = state[getState(StateType::SCH_Type)].curTime();
    amrex::Real dt = parent->dtLevel(0);
    std::string FileName = dir + "/comoving_a";
    std::ofstream File;
    File.open(FileName.c_str(), std::ios::out | std::ios::trunc);
    if (!File.good()) {
      amrex::FileOpenFailed(FileName);
    }
    File.precision(15);
    File << Comoving::get_comoving_a(time) << '\n';
    // File << Comoving::get_comoving_a(time - dt) << '\n';
    File << time << '\n';
    // File << time - dt << '\n';
    File.close();
  }
}
