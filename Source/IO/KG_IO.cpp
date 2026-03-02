#include <AxKG.H>

// #include <DFFTUtils.H>

#include <unistd.h>

extern std::string inputs_name;

AxKG::ICType AxKG::ic = AxKG::ICType::uniform;
int AxKG::simType = -1;
amrex::Real AxKG::KG0 = -1;       // Need the initial value of the field.  --PH
amrex::Real AxKG::KG_k = -1;      // The chosen momentum for fixed_k type ICs
amrex::Real AxKG::cutoff_k = -1;  // Momentum cutoff for momentum-space ICs 
amrex::Real AxKG::A = -1;
amrex::Real AxKG::B = -1;
amrex::Real AxKG::r = -1;
amrex::Real AxKG::beta = -1;
amrex::Real AxKG::s = -1;
std::vector<amrex::Real> AxKG::simPars;

// Read in the parameters specific to a KG run
void AxKG::read_params ()
{
    BaseAx::read_params();

    BL_PROFILE("AxKG::read_params()");

    amrex::ParmParse pp_KG("KG");

    // Code units
    pp_KG.get("KG0", KG0);
    if (pp_KG.contains("A"))
        pp_KG.query("A", A);
    else
        A = 1./KG0;
    if (pp_KG.contains("beta"))
		pp_KG.query("beta", beta);
    if (pp_KG.contains("B"))
		pp_KG.query("B", B);

    // Simulation model-specific parameters
    pp_KG.query("simType", simType); // Extract number of fields
    switch (simType)
    {
        case SFQ:
            simPars.resize(2);
            pp_KG.get("mass", simPars[0]);
            pp_KG.get("lambda", simPars[1]);
            if (!pp_KG.contains("beta"))
            {
                if(simPars[0] == 0.)
                    beta = 4.; 
                else
                    beta = 2.;  
            }
            if (!pp_KG.contains("B"))
            {
                if(simPars[0] == 0.)
                    B = sqrt(abs(simPars[1]))*pow(KG0, -1. + beta/2.);
                else
                    B = simPars[0];
            }
            break;
        case GMON:
            simPars.resize(4);
            pp_KG.get("mass", simPars[0]);   // m
            pp_KG.get("MASS", simPars[1]);   // M
            pp_KG.get("power", simPars[2]);  // alpha
            if (pp_KG.contains("d"))
            {
                pp_KG.get("d",simPars[3]); // phi^2/M^2 should be largish during inflation, so the dominant term should be (phi^2/M^2)^alpha
            }
            else
            {
                simPars[3] = 0.;
            }
            if (!pp_KG.contains("beta"))
            {
                beta = 2.*simPars[2]; // phi^2/M^2 should be largish during inflation, so the dominant term should be (phi^2/M^2)^alpha
                if(simPars[3] != 0.)
                    beta = 4.;        // ... unless we're using the reduced potential, in which case the dominant term for large phi will be quartic.
            }
            if (!pp_KG.contains("B"))
            {
                    //  m * M^(1-alpha) * phi0^(-1+beta/2)
                B = simPars[0]*pow(simPars[1],1.-simPars[2])*pow(KG0, -1. + beta/2.);
                if(simPars[3] != 0.)  // ... unless we're reduced, in which case sqrt(cpl) = sqrt(1 - alpha) m / (2 M)
                    B = sqrt(abs(1. - simPars[2])*simPars[3])*simPars[0]*pow(KG0, -1. + beta/2.)/(2.*simPars[1]);
            }
            break;
        case LOR:
            simPars.resize(2);
            pp_KG.get("mass", simPars[0]);   // m
            pp_KG.get("MASS", simPars[1]);   // M
            if (!pp_KG.contains("beta"))
            {
                // beta = 0; // Lorentzian goes to a constant for large field values, V = 0.5 m^2 phi^2 / (1 + 0.5 m^2 phi^2 / M^4) -> M^2
                beta = 2; // Lorentzian goes to a phi^2 for small field values, V = 0.5 m^2 phi^2 / (1 + 0.5 m^2 phi^2 / M^4) -> 0.5 m^2 phi^2
            }
            if (!pp_KG.contains("B"))
            {
                //  M^2 * phi0^(-1+beta/2)
                // B = simPars[1]*simPars[1]*pow(KG0, -1. + beta/2.);
                //  m * phi0^(-1+beta/2)
                B = simPars[0]*pow(KG0, -1. + beta/2.);
            }
            break;
        case TMI:
            simPars.resize(3);
            pp_KG.get("mass", simPars[0]);   // m
            pp_KG.get("alpha", simPars[1]);   // alpha
            pp_KG.get("n_pow", simPars[2]);   // n_pow
            if (!pp_KG.contains("beta"))
            {
                beta = 2 * simPars[2];  
            }
            if (!pp_KG.contains("B"))
            {
                //  M^2 * phi0^(-1+beta/2)
                // B = simPars[1]*simPars[1]*pow(KG0, -1. + beta/2.);
                //  m * phi0^(-1+beta/2)
                B = sqrt(simPars[0]*simPars[0]*simPars[2])*pow(KG0 / sqrt(6.*simPars[1]), -1 + simPars[2]);
            }
            break;
        case EMI:
            simPars.resize(3);
            pp_KG.get("mass", simPars[0]);   // m
            pp_KG.get("alpha", simPars[1]);   // alpha
            pp_KG.get("n_pow", simPars[2]);   // n_pow
            if (!pp_KG.contains("beta"))
            {
                beta = 2 * simPars[2];  
            }
            if (!pp_KG.contains("B"))
            {
                //  M^2 * phi0^(-1+beta/2)
                // B = simPars[1]*simPars[1]*pow(KG0, -1. + beta/2.);
                //  m * phi0^(-1+beta/2)
                B = sqrt(2*simPars[2] * simPars[1] * pow(2. / (3. * simPars[1]), simPars[2]))*simPars[0]*pow(KG0, -1 + simPars[2]);
            }
            break;

        default:
            amrex::Abort("KG_IO::read_params() Error: simType is not a recognized simulation type.");
    }

    // Initial condition type is required
    int intIC; 
    pp_KG.get("ICType", intIC);
    ic = getIC(intIC);

    if(ic == ICType::delta_k || ic == ICType::standard)
    {
        if (pp_KG.contains("cutoff_k"))
        {
            pp_KG.get("cutoff_k", cutoff_k);
        }
        else
        {
            cutoff_k = 0;
        }
    }

    if (pp_KG.contains("KG_k"))
        pp_KG.query("KG_k", KG_k);
    else
        KG_k = 1;

    // Remaining code unit variables
    if(pp_KG.contains("r"))
        pp_KG.query("r", r);
    else
        r = 6./(2. + beta);
    if(pp_KG.contains("s"))
        pp_KG.query("s", s);
    else
        s = 3.*(2. - beta)/(2. + beta);
}

void AxKG::write_info ()
{
	int ndatalogs = parent->NumDataLogs();
    amrex::Real time_unit = 1.0; //3.0856776e19 / 31557600.0; // conversion to Julian years

	int rlp = AxKG::runlog_precision;

	if (ndatalogs > 0)
	{

        amrex::Real time = state[getState(StateType::KG_Type)].curTime(); 
		amrex::Real dt    = parent->dtLevel(0);
		int  nstep = parent->levelSteps(0);
        
        //// Print the average value of the field and its derivative so we can make a phase-space plot
        int gridsize = geom.Domain().length(0)*geom.Domain().length(1)*geom.Domain().length(2); // Assuming a rectangular domain

        std::unique_ptr<amrex::MultiFab> Phi = AxKG::derive("KGf", time, 0);
        std::unique_ptr<amrex::MultiFab> dPhi = AxKG::derive("KGfv", time, 0);

        amrex::Real avPhi = Phi->sum()/gridsize;
        amrex::Real avdPhi = dPhi->sum()/gridsize;
        ///////

		if (amrex::ParallelDescriptor::IOProcessor())
		{
			std::ostream& data_loga = parent->DataLog(0);
            if (time == 0.0)
            {
                data_loga << std::setw( 8) <<  "#  nstep";
                data_loga << std::setw(14) <<  "       time    ";
                data_loga << std::setw(14) <<  "       dt      ";
                data_loga << std::setw(14) <<  "       <phi>     ";
                data_loga << std::setw(14) <<  "       <\\dot phi>     ";
                data_loga << std::endl;

            }
            data_loga << std::setw( 8) <<  nstep;
            data_loga << std::setw(14) <<  std::setprecision(rlp) <<  time * time_unit;
            data_loga << std::setw(14) <<  std::setprecision(rlp) <<    dt * time_unit;
            data_loga << std::setw(14) <<  std::setprecision(rlp) <<    avPhi;
            data_loga << std::setw(14) <<  std::setprecision(rlp) <<    avdPhi;
            data_loga << std::endl;
		}
	}
}

void AxKG::writePlotFilePost (const std::string& dir, std::ostream& os)
{
    BaseAx::writePlotFilePost(dir, os);

    outputPowerSpectrum(dir);

}

void AxKG::outputPowerSpectrum(const std::string& dir)
{
    // Spits out the power spectrum for phi to a text file in the plot folder.

    const int L = Domain().length(0);

    amrex::Real time = state[getState(StateType::KG_Type)].curTime(); 
    // const int nbins = std::floor(std::sqrt(Domain().length(0)*Domain().length(0) + Domain().length(1)*Domain().length(1) + Domain().length(2)*Domain().length(2)));
    const int nbins = std::floor(std::sqrt(3)*L);
    amrex::Vector<amrex::Real> spectrum(nbins,0.0);
    amrex::Vector<amrex::Real> mode(nbins,0.0);
    amrex::Vector<int>  nmode(nbins,0);
    amrex::Vector<amrex::Real> spectrumx((Domain().length(0)),0.0);
    amrex::Vector<amrex::Real> modex((Domain().length(0)),0.0);
    const static amrex::Real len = geom.ProbHi(0)-geom.ProbLo(0);

    // const auto geomdata = geom.data();

    std::unique_ptr<amrex::MultiFab> Phi = static_cast<AxKG*>(&get_level(0))->derive("KGf", time, 0);

    static amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::forward> fft(geom.Domain());
    auto const& [ba,dm] = fft.getSpectralDataLayout();

    amrex::cMultiFab phi_fft(ba, dm, 1, 0);

    fft.forward(*Phi, phi_fft);

    // // phi_fft = DFFTUtils::forward_dfft(*Phi, geomdata, 0, false);
    // phi_fft = DFFTUtils::forward_dfft(*Phi, 0, false);
    // // phi_fft.mult(1./phi_fft.boxArray().numPts());

    for (amrex::MFIter mfi(phi_fft,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx  = mfi.tilebox();
        const auto fab = phi_fft.array(mfi);
        amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) noexcept  // Note the & instead of =!! This takes references for external variables as opposed to making local copies.
        {
            int ki = i;
            // int kj = (j <= L/2) ? j : L - j;
            // int kk = (k <= L/2) ? k : L - k;
            int kj =  j ;
            int kk =  k ;
	    const int bin = std::floor(std::sqrt(ki*ki + kj*kj + kk*kk));

	    nmode[bin]++;
            spectrum[bin]  += amrex::norm(fab(ki,kj,kk,0));
	    mode[bin]      += std::sqrt(ki*ki + kj*kj + kk*kk)*M_PI/len;
            spectrumx[ki]  += amrex::norm(fab(ki,kj,kk,0));
            modex[ki]       = ki;
        });
    }

    amrex::ParallelDescriptor::ReduceIntSum(nmode.dataPtr(), nmode.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(spectrum.dataPtr(), spectrum.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(mode.dataPtr(), mode.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(spectrumx.dataPtr(), spectrumx.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(modex.dataPtr(), modex.size(),
                amrex::ParallelDescriptor::IOProcessorNumber());

    amrex::ParallelDescriptor::Barrier();
      
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::string fileName = dir + "/spectrum";
        std::ofstream spectrum_out(fileName.c_str());
        for(int bin=0; bin < spectrum.size(); ++bin) {
          spectrum_out << mode[bin] << " " << spectrum[bin] << " " << nmode[bin] << std::endl;
        }
        spectrum_out.close();
        std::string fileNamex = dir + "/spectrumx";
        std::ofstream spectrumx_out(fileNamex.c_str());
        for(int i=0; i < modex.size(); ++i) {
          spectrumx_out << modex[i] << " " << spectrumx[i]/(128.*128.) << std::endl;
        }
        spectrumx_out.close();
    }

}
