//#include <AxNewt.H>
//#include <Newtonian.H>
//#include <constants_cosmo.H> // Include cosmological constants
//#include <BaseAx.H>

//#include <bc_fill.H>
//#include <istream>
//#include <ostream>
//#include <unistd.h>

#include <AxNewt.H>
#include <Comoving_Full.H>

#include <istream>
#include <ostream>
#include <unistd.h>

void AxNewt::write_info() {
  int ndatalogs = parent->NumDataLogs();
  amrex::Real time_unit = 1.0;
  int rlp = AxNewt::runlog_precision;

  if (ndatalogs > 0) {
    amrex::Real time = state[getState(StateType::State_Type)]
                           .curTime(); // PhiGrav_Type? Gravity_Type?
    amrex::Real dt = parent->dtLevel(0);
    int nstep = parent->levelSteps(0);

    int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                   geom.Domain().length(2);

    amrex::MultiFab &densitygrav_old =
        get_level(level).get_new_data(getState(StateType::State_Type));
    amrex::MultiFab &phigrav_old =
        get_level(level).get_new_data(getState(StateType::PhiGrav_Type));
    amrex::MultiFab &gradphi_old =
        get_level(level).get_new_data(getState(StateType::Gravity_Type));

    amrex::Real avdensity = densitygrav_old.sum() / gridsize;
    amrex::Real avphigrav = phigrav_old.sum() / gridsize;
    amrex::Real avgradphi = gradphi_old.sum() / gridsize;

    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ostream &data_log = parent->DataLog(0);
      if (time == 0.0) {
        data_log << std::setw(8) << "#  nstep";
        data_log << std::setw(14) << "          time";
        data_log << std::setw(14) << "         dt";
        data_log << std::setw(14) << "  <Density>";
        data_log << std::setw(14) << "     <PhiGrav>";
        data_log << std::setw(14) << " <|GradPhiGrav|>";
        data_log << std::endl;
      }
      data_log << std::setw(8) << nstep;
      data_log << std::setw(14) << std::setprecision(rlp) << time * time_unit;
      data_log << std::setw(14) << std::setprecision(rlp) << dt * time_unit;
      data_log << std::setw(14) << std::setprecision(rlp) << avdensity;
      data_log << std::setw(14) << std::setprecision(rlp) << avphigrav;
      data_log << std::setw(14) << std::setprecision(rlp) << avgradphi;
      data_log << std::endl;
    }
  }

  amrex::Print() << "checkpoint Newt_IO::write_info\n";
}

void AxNewt::writePlotFilePost(const std::string &dir, std::ostream &os) {
  BaseAx::writePlotFilePost(dir, os); // LSR -- run BaseAx version first
  if (write_skip_prepost != 1) {
    if (verbose) {
      if (level == 0) {
        amrex::Print().SetPrecision(15)
            << "Output file " << dir << " at scale factor "
            << std::to_string(Comoving::get_comoving_a()) << " and step "
            << std::to_string(nStep()) << std::endl;
      }
    }
  }
//  outputPotentialSpectrum(dir);

  amrex::Print() << "checkpoint Newt_IO::writePlotFilePost\n";
}

// void AxNewt::outputPotentialSpectrum(const std::string &dir) {}
