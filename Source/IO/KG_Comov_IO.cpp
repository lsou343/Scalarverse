#include <AxKGComov.H>
#include <Comoving_Full.H>

#include <istream>
#include <ostream>
#include <unistd.h>

// Read in the parameters specific to a KG run
void AxKGComov::read_params() { AxKG::read_params(); }

void AxKGComov::write_info() {
  int ndatalogs = parent->NumDataLogs();
  amrex::Real time_unit =
      1.0; // 3.0856776e19 / 31557600.0; // conversion to Julian years

  int rlp = AxKG::runlog_precision;

  if (ndatalogs > 0) {

    amrex::Real time = state[getState(StateType::KG_Type)].curTime();
    amrex::Real dt = parent->dtLevel(0);
    int nstep = parent->levelSteps(0);

    //// N.B. This needs to be updated for refined data
    //// Print the average value of the field and its derivative
    int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                   geom.Domain().length(2); // Assuming a rectangular domain

    std::unique_ptr<amrex::MultiFab> Phi = AxKG::derive("KGf", time, 0);
    std::unique_ptr<amrex::MultiFab> dPhi = AxKG::derive("KGfv", time, 0);

    amrex::Real avPhi = Phi->sum() / gridsize;
    amrex::Real avdPhi = dPhi->sum() / gridsize;

    int flev = parent->finestLevel();
    while (parent->getAmrLevels()[flev] == nullptr)
      flev--;
    ///////

    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ostream &data_loga = parent->DataLog(0);
      if (time == 0.0) {
        data_loga << std::setw(8) << "#  nstep";
        data_loga << std::setw(14) << "       time    ";
        data_loga << std::setw(14) << "       dt      ";
        data_loga << std::setw(14) << "       a       ";
        data_loga << std::setw(14) << "       ap     ";
        data_loga << std::setw(14) << "       app     ";
        data_loga << std::setw(14) << "       e-fld     ";
        data_loga << std::setw(14) << "       <phi>     ";
        data_loga << std::setw(14) << "       <\\dot phi>     ";
        data_loga << std::setw(14) << "       H^2/(8 pi rho/3)     ";
        data_loga << std::setw(14) << "       flev     ";
        data_loga << std::endl;
      }
      data_loga << std::setw(8) << nstep;
      data_loga << std::setw(14) << std::setprecision(rlp) << time * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp) << dt * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << Comoving::get_comoving_a();
      data_loga << std::setw(14) << std::setprecision(rlp)
                << Comoving::get_comoving_ap();
      data_loga << std::setw(14) << std::setprecision(rlp)
                << Comoving::get_comoving_app();
      data_loga << std::setw(14) << std::setprecision(rlp)
                << std::log(Comoving::get_comoving_a());
      data_loga << std::setw(14) << std::setprecision(rlp) << avPhi;
      data_loga << std::setw(14) << std::setprecision(rlp) << avdPhi;
      data_loga << std::setw(14) << std::setprecision(rlp)
                << Comoving::debug_ratio();
      data_loga << std::setw(14) << std::setprecision(rlp) << flev;
      data_loga << std::endl;

      if (ndatalogs > 1) {
        std::ostream &data_logb = parent->DataLog(1);
        if (time == 0.0) {
          data_logb << std::setw(8) << "#  nstep";
          data_logb << std::setw(14) << "       time    ";
          data_logb << std::setw(14) << "       rho_t    ";
          data_logb << std::setw(14) << "       rho_g    ";
          data_logb << std::setw(14) << "       rho_v    ";
          data_logb << std::setw(14) << "       H^2/(8 pi rho/3)     ";
          data_logb << std::setw(14) << "       eos_w    ";
          data_logb << std::setw(14) << "       flev     ";
          data_logb << std::endl;
        }
        data_logb << std::setw(8) << nstep;
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << time * time_unit;
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << Comoving::get_rho_t() / Comoving::get_gridsize();
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << Comoving::get_rho_g() / Comoving::get_gridsize();
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << Comoving::get_rho_v() / Comoving::get_gridsize();
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << Comoving::debug_ratio();
        data_logb << std::setw(14) << std::setprecision(rlp)
                  << (Comoving::get_rho_t() - Comoving::get_rho_v()) /
                         (Comoving::get_rho_t() + Comoving::get_rho_v());

        data_logb << std::setw(14) << std::setprecision(rlp) << flev;
        data_logb << std::endl;
      }
    }
  }
}

void AxKGComov::writePlotFilePost(const std::string &dir, std::ostream &os) {
  AxKG::writePlotFilePost(dir, os);

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
}

void AxKGComov::checkPointPost(const std::string &dir, std::ostream &os) {

  // Write comoving_a into its own file in the checkpoint directory
  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Real time = state[getState(StateType::KG_Type)].curTime();
    amrex::Real dt = parent->dtLevel(0);
    std::string FileName = dir + "/comoving_a";
    std::ofstream File;
    File.open(FileName.c_str(), std::ios::out | std::ios::trunc);
    if (!File.good()) {
      amrex::FileOpenFailed(FileName);
    }
    File.precision(15);
    File << Comoving::get_comoving_a(time) << '\n';
    File << Comoving::get_comoving_ap(time) << '\n';
    File << Comoving::get_comoving_app(time) << '\n';
    File << Comoving::get_comoving_a(time - dt) << '\n';
    File << Comoving::get_comoving_ap(time - dt) << '\n';
    File << Comoving::get_comoving_app(time - dt) << '\n';
    File << time << '\n';
    File << time - dt << '\n';
    File.close();
  }
}
