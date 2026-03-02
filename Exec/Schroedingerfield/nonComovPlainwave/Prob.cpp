#include <AxSCH.H>
using namespace amrex;
void AxSCH::prob_initdata_pos(
    const int i, const int j, const int k,
    amrex::Array4<amrex::Real> const &fields,
    amrex::GeometryData const &geomdata,
    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {

  static AxSCH::ICType ic = AxSCH::getIC(prob_param[0]);

  switch (ic) {
  case AxSCH::ICType::test:
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) = 1.;
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) = 1.;
    // do not forget to set initial density and phase otherwise
    // Nan's will appear in the MultiFab
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Dens)) = 1.;
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Phase)) = 0.;
    break;
  case AxSCH::ICType::plain_wave:
    SCH0 = prob_param[1];
    SCH_k0 = prob_param[2];
    Phase0 = prob_param[3];
    if (SCH0 == PAR_ERR_VAL || SCH_k0 == PAR_ERR_VAL || Phase0 == PAR_ERR_VAL)
      amrex::Error("prob_initdata_state: in plain_wave IC---insufficient "
                   "number of problem parameters!");
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) =
        SCH0 *
        std::cos(SCH_k0 * (geomdata.ProbLo(0) + i * geomdata.CellSize(0)) * 2. *
                     M_PI / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) +
                 Phase0); // Needs to be 2pi so we have a full period to satisfy
                          // smooth periodic boundary conditions.
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) =
        SCH0 *
        std::sin(SCH_k0 * (geomdata.ProbLo(0) + i * geomdata.CellSize(0)) * 2. *
                     M_PI / (geomdata.ProbHi(0) - geomdata.ProbLo(0)) +
                 Phase0);
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Dens)) =
        fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) *
            fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) +
        fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) *
            fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im));
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Phase)) =
        std::atan2(fields(i, j, k, AxSCH::getField(Fields::SCHf_Im)),
                   fields(i, j, k, AxSCH::getField(Fields::SCHf_Re)));
    ;
    break;

    /*        case AxSCH::ICType::uniform: */
    /*if(prob_param[1] == PAR_ERR_VAL)*/
    /*amrex::Error("prob_initdata_state: in uniform IC---insufficient number of
     * problem parameters!");*/
    /*fields(i,j,k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) = 1.; // F_pr = a^r
     * f/f_0, so F_pr(t = 0) = 1.*/
    /*fields(i,j,k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) = 1.;*/
    /*//fields(i,j,k, AxSCH::getField(AxSCH::Fields::SCHfv)) = 0.;*/
    /*break;*/
    /*case AxSCH::ICType::fixed_k:*/
    /*if(prob_param[1] == PAR_ERR_VAL || prob_param[2] == PAR_ERR_VAL)*/
    /*amrex::Error("prob_initdata_state: in fixed_k IC---insufficient number of
     * problem parameters!");*/
    /*SCH0 = prob_param[1];*/
    /*static amrex::Real SCH_k = prob_param[2];*/
    // fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf)) =
    // SCH0*std::cos(SCH_k*(geomdata.ProbLo(0) +
    // i*geomdata.CellSize(0))*2.*M_PI/(geomdata.ProbHi(0) -
    // geomdata.ProbLo(0))); // Needs to be 2pi so we have a full period to
    // satisfy smooth periodic boundary conditions. fields(i, j, k,
    // AxSCH::getField(AxSCH::Fields::SCHf)) = 1. +
    // std::cos(SCH_k*(geomdata.ProbLo(0) +
    // i*geomdata.CellSize(0))*2.*M_PI/(geomdata.ProbHi(0) -
    // geomdata.ProbLo(0))); // Needs to be 2pi so we have a full period to
    // satisfy smooth periodic boundary conditions.
    // fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHfv)) = 0.;
    // break;
  case AxSCH::ICType::gaussian: {
    // Retrieve Gaussian parameters.
    Real amplitude = prob_param[1];
    Real sigma = prob_param[2]; // Add sigma as an input parameter
    Real k0 = prob_param[3];    // Initial wave number (momentum)

    // Set default center to the middle of the domain in each direction.
    Real centerX =
        geomdata.ProbLo(0) + 0.5 * (geomdata.ProbHi(0) - geomdata.ProbLo(0));
    Real centerY =
        geomdata.ProbLo(1) + 0.5 * (geomdata.ProbHi(1) - geomdata.ProbLo(1));
    Real centerZ =
        geomdata.ProbLo(2) + 0.5 * (geomdata.ProbHi(2) - geomdata.ProbLo(2));

    // Compute physical coordinates at the current cell.
    Real x = geomdata.ProbLo(0) + i * geomdata.CellSize(0);
    Real y = geomdata.ProbLo(1) + j * geomdata.CellSize(1);
    Real z = geomdata.ProbLo(2) + k * geomdata.CellSize(2);

    // Calculate squared distance from the center.
    Real r2 = (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY) +
              (z - centerZ) * (z - centerZ);

    // Evaluate the Gaussian envelope.
    Real gauss = amplitude * std::exp(-r2 / (2.0 * sigma * sigma));

    // Add initial wave oscillation with wave number k0.
    Real L_box = geomdata.ProbHi(0) - geomdata.ProbLo(0);
    Real k_wavenumber =
        (2.0 * M_PI / L_box) * k0; // Convert mode number to wavenumber
    Real phase = k_wavenumber * x; // Wave propagates in x-direction

    // Initialize real and imaginary parts.
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) =
        gauss * std::cos(phase);
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) =
        gauss * std::sin(phase);

    // Set density as the squared modulus.
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Dens)) =
        fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) *
            fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)) +
        fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)) *
            fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im));

    // Set phase (atan2 ensures correct quadrant).
    fields(i, j, k, AxSCH::getField(AxSCH::Fields::Phase)) =
        std::atan2(fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Im)),
                   fields(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re)));

    break;
  }

  default:
    amrex::Error("prob_initdata_state: Selected initial condition type is not "
                 "yet implemented!");
  }
  // {
  /* Real rando = Random(); */
  /* field(i,j,k,AxSCH::SCHf)   = LatticeEasy stuff */
  /* field(i,j,k,AxSCH::SCHfv)   = LatticeEasy stuff */
  // }
}

#ifdef BL_USE_MPI
// initial condition at i,j,k in momentum-space. "state" contains field data (to
// be initialized here), "geomdata" contains the problem geometry, and
// "prob_param" contains the relevant problem parameters.
void AxSCH::prob_initdata_mom(
    const int i, const int j, const int k,
    amrex::Array4<amrex::GpuComplex<amrex::Real>> const
        &fields, // Change Real -> GpuComplex<Real>
    amrex::GeometryData const &geomdata,
    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {
  // For momentum-space ICs we have both real and imaginary parts for each
  // field, so the usual indices from initial condition (ic) standard and
  // delta_k are not implemented yet
  static AxSCH::ICType ic = AxSCH::getIC(prob_param[0]);

  switch (ic) {
    /* case AxSCH::ICType::delta_k:*/
    /*//print that this is not implemented yet*/
    /*amrex::Error("prob_initdata_mom: in delta_k IC---not yet implemented!");*/
  default:
    amrex::Error("prob_initdata_mom: Selected initial condition type is not "
                 "yet implemented!");
  }
}
#endif // BL_USE_MPI
