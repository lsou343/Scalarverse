#include <AxSCH.H>
#ifdef BL_USE_MPI
// #include <DFFTUtils.H>
#endif // BL_USE_MPI

#include <unistd.h>

extern std::string inputs_name;

AxSCH::ICType AxSCH::ic = AxSCH::ICType::test;
int AxSCH::PSorFD = -1;         // 0 = PS, 1 = FD
int AxSCH::PSorder = -1;        // Order of the PS method
amrex::Real AxSCH::SCH0 = -1;   // Need the initial value of the field.  --PH
amrex::Real AxSCH::SCH_k0 = 0;  // The chosen momentum for fixed_k type ICs
amrex::Real AxSCH::Phase0 = -1; // Initial phase of the field.
amrex::Real AxSCH::sigma = 0.1; // Width of the Gaussian
amrex::Real AxSCH::vonNeumann_dt = 0.0; // Von Neumann stability factor
amrex::Real AxSCH::mtt = -1;            // 2.5; // Mass of the scalar field
amrex::Real AxSCH::hbaroverm = -1;      // Reduced Planck constant over mass
amrex::Real AxSCH::test_factor =
    1.0; // Factor for testing, that way one does not need to comile every time

// If Initialized from KG field:
std::string AxSCH::KGinitDirName =
    "No KGinitDirName given"; // Directory name for MultiFab
amrex::Real AxSCH::KGm = -1;
amrex::Real AxSCH::KGA = -1;
amrex::Real AxSCH::KGB = -1;
amrex::Real AxSCH::KGr = -1;
amrex::Real AxSCH::KGs = -1;

// Read in the parameters specific to a SCH run
void AxSCH::read_params() {
  BaseAx::read_params();

  BL_PROFILE("AxSCH::read_params()");

  amrex::ParmParse pp_SCH(
      "SCH"); // Select values of input file with prefix "SCH"
  pp_SCH.get("PSorFD", PSorFD);
  pp_SCH.query("PSorder", PSorder);
  pp_SCH.query("test_factor", test_factor);
  pp_SCH.query("SCH0", SCH0);
  pp_SCH.query("Phase0", Phase0);
  pp_SCH.get("vonNeumann_dt", vonNeumann_dt);
  pp_SCH.get("mtt", mtt);
  pp_SCH.get(
      "hbaroverm",
      hbaroverm); // amrex::Real AxSCH::hbaroverm = 0.01917152 / mtt; // hbar/m
  /*    if (pp_SCH.contains("A"))*/
  /*pp_SCH.query("A", A);*/
  /*else*/
  // A = 1./SCH0;
  /*  if (pp_SCH.contains("B"))*/
  /*pp_SCH.query("B", B);*/

  // Initial condition type is required
  int intIC;
  pp_SCH.get("ICType", intIC);
  ic = getIC(intIC);

  if (pp_SCH.contains("SCH_k0"))
    pp_SCH.query("SCH_k0", SCH_k0);
  else
    SCH_k0 = 0;

  if (pp_SCH.contains("sigma"))
    pp_SCH.query("sigma", sigma);
  else
    sigma = 0.1;

  amrex::ParmParse pp_KG("KG");
  // if Initialized from KG field
  if (ic == ICType::KGfield) {
    pp_KG.get("KGinitDirName", KGinitDirName);
    // TODO read alpha and beta instead of A and B
    pp_KG.get("KGm", KGm);
    pp_KG.get("KGA", KGA);
    pp_KG.get("KGB", KGB);
    pp_KG.get("KGr", KGr);
    pp_KG.get("KGs", KGs);
  }
}

void AxSCH::write_info() {
  int ndatalogs = parent->NumDataLogs();
  amrex::Real time_unit =
      1.0; // 3.0856776e19 / 31557600.0; // conversion to Julian years

  int rlp = AxSCH::runlog_precision;

  if (ndatalogs > 0) {

    amrex::Real time = state[getState(StateType::SCH_Type)].curTime();
    amrex::Real dt = parent->dtLevel(0);
    int nstep = parent->levelSteps(0);

    //// Print the average value of the field and its derivative so we can make
    /// a phase-space plot
    int gridsize = geom.Domain().length(0) * geom.Domain().length(1) *
                   geom.Domain().length(2); // Assuming a rectangular domain

    std::unique_ptr<amrex::MultiFab> Psi_Re = AxSCH::derive("SCHf_Re", time, 0);
    std::unique_ptr<amrex::MultiFab> Psi_Im = AxSCH::derive("SCHf_Im", time, 0);
    std::unique_ptr<amrex::MultiFab> rho = AxSCH::derive("Dens", time, 0);

    amrex::Real avPsi_Re = Psi_Re->sum() / gridsize;
    amrex::Real avPsi_Im = Psi_Im->sum() / gridsize;
    amrex::Real avrho = rho->sum() / gridsize;
    ///////

    if (amrex::ParallelDescriptor::IOProcessor()) {
      std::ostream &data_loga = parent->DataLog(0);
      if (time == 0.0) {
        data_loga << std::setw(8) << "      nstep";
        data_loga << std::setw(14) << "   time    ";
        data_loga << std::setw(14) << "     dt      ";
        data_loga << std::setw(14) << "  <Psi_Re>     ";
        data_loga << std::setw(14) << " <Psi_Im>     ";
        data_loga << std::setw(14) << "     <rho>     ";
        data_loga << std::endl;
      }
      data_loga << std::setw(8) << nstep;
      data_loga << std::setw(14) << std::setprecision(rlp) << time * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp) << dt * time_unit;
      data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Re;
      data_loga << std::setw(14) << std::setprecision(rlp) << avPsi_Im;
      data_loga << std::setw(14) << std::setprecision(rlp) << avrho;
      data_loga << std::endl;
    }
  }
}

void AxSCH::writePlotFilePost(const std::string &dir, std::ostream &os) {
  BaseAx::writePlotFilePost(dir, os);

  outputPowerSpectrum(dir);
}

void AxSCH::outputPowerSpectrum(const std::string &dir) {
  // Spits out the power spectrum for phi to a text file in the plot folder
  // (best solution I've got so far).

  amrex::Real time = state[getState(StateType::SCH_Type)].curTime();
  const int nbins =
      std::floor(std::sqrt(Domain().length(0) * Domain().length(0) +
                           Domain().length(1) * Domain().length(1) +
                           Domain().length(2) * Domain().length(2)));
  amrex::Vector<amrex::Real> spectrum(nbins, 0.0);
  amrex::Vector<amrex::Real> mode(nbins, 0.0);
  amrex::Vector<int> nmode(nbins, 0);
  amrex::Vector<amrex::Real> spectrumx((Domain().length(0)), 0.0);
  amrex::Vector<amrex::Real> modex((Domain().length(0)), 0.0);
  const static amrex::Real len = geom.ProbHi(0) - geom.ProbLo(0);

  const auto geomdata = geom.data();

  std::unique_ptr<amrex::MultiFab> Phi =
      static_cast<AxSCH *>(&get_level(0))->derive("SCHf_Re", time, 0);
  amrex::MultiFab phi_fft;

  // phi_fft = DFFTUtils::forward_dfft(*Phi, geomdata, 0, false);
  // phi_fft = DFFTUtils::forward_dfft(*Phi, 0, false);
  // phi_fft.mult(1./phi_fft.boxArray().numPts());

  for (amrex::MFIter mfi(phi_fft, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const amrex::Box &bx = mfi.tilebox();
    const auto fab = phi_fft.array(mfi);
    amrex::ParallelFor(
        bx, [&] AMREX_GPU_DEVICE(
                int i, int j,
                int k) noexcept // Note the & instead of =!! This takes
                                // references for external variables as opposed
                                // to making local copies.
        {
          const int bin = std::floor(std::sqrt(i * i + j * j + k * k));

          nmode[bin]++;
          spectrum[bin] += fab(i, j, k, 0) * fab(i, j, k, 0) +
                           fab(i, j, k, 1) * fab(i, j, k, 1);
          mode[bin] += std::sqrt(i * i + j * j + k * k) * M_PI / len;
          spectrumx[i] += fab(i, j, k, 0) * fab(i, j, k, 0) +
                          fab(i, j, k, 1) * fab(i, j, k, 1);
          modex[i] = i;
        });
  }

  amrex::ParallelDescriptor::ReduceIntSum(
      nmode.dataPtr(), nmode.size(),
      amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::ReduceRealSum(
      spectrum.dataPtr(), spectrum.size(),
      amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::ReduceRealSum(
      mode.dataPtr(), mode.size(),
      amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::ReduceRealSum(
      spectrumx.dataPtr(), spectrumx.size(),
      amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::ReduceRealSum(
      modex.dataPtr(), modex.size(),
      amrex::ParallelDescriptor::IOProcessorNumber());

  amrex::ParallelDescriptor::Barrier();

  if (amrex::ParallelDescriptor::IOProcessor()) {
    std::string fileName = dir + "/spectrum";
    std::ofstream spectrum_out(fileName.c_str());
    for (int bin = 0; bin < spectrum.size(); ++bin) {
      spectrum_out << mode[bin] << " " << spectrum[bin] << " " << nmode[bin]
                   << std::endl;
    }
    spectrum_out.close();
    std::string fileNamex = dir + "/spectrumx";
    std::ofstream spectrumx_out(fileNamex.c_str());
    for (int i = 0; i < modex.size(); ++i) {
      spectrumx_out << modex[i] << " " << spectrumx[i] / (128. * 128.)
                    << std::endl;
    }
    spectrumx_out.close();
  }
}
