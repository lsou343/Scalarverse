// LSR -- taken from ComovSingleField
// #include <Prob.H>
#include <AxKG.H>
#include <KG_compute_models.H>

#ifdef COMOV_FULL
#include <Comoving_Full.H>
#endif

#define randa 16807
#define randm 2147483647
#define randq 127773
#define randr 2836
float rand_uniform(void) {
  static int i = 0;
  static int next = 1.0;
  if (!(next > 0)) // Guard against 0, negative, or other invalid seeds
  {
    printf("Invalid seed used in random number function. Using seed=1\n");
    next = 1;
  }
  if (i == 0) // On the first call run through 100 calls. This allows small
              // seeds without the first results necessarily being small.
    for (i = 1; i < 100; i++)
      rand_uniform();
  next = randa * (next % randq) - randr * (next / randq);
  if (next < 0)
    next += randm;
  return ((float)next / (float)randm);
}
#undef randa
#undef randm
#undef randq
#undef randr

// initial condition at i,j,k in position-space. "state" contains field data (to
// be initialized here), "geomdata" contains the problem geometry, and
// "prob_param" contains the relevant problem parameters.
void AxKG::prob_initdata_pos(
    const int i, const int j, const int k,
    amrex::Array4<amrex::Real> const &fields,
    amrex::GeometryData const &geomdata,
    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {

  static AxKG::ICType ic = AxKG::getIC(prob_param[0]);

  // Can start with a super simple homogeneous field. --PH
  switch (ic) {
  case AxKG::ICType::uniform:
    if (prob_param[1] == PAR_ERR_VAL)
      amrex::Error("prob_initdata_state: in uniform IC---insufficient number "
                   "of problem parameters!");
    fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) =
        1.; // F_pr = a^r f/f_0, so F_pr(t = 0) = 1.
    fields(i, j, k, AxKG::getField(AxKG::Fields::KGfv)) = 0.;
    break;
  case AxKG::ICType::fixed_k:
    if (prob_param[1] == PAR_ERR_VAL || prob_param[2] == PAR_ERR_VAL)
      amrex::Error("prob_initdata_state: in fixed_k IC---insufficient number "
                   "of problem parameters!");
    KG0 = prob_param[1];
    static amrex::Real KG_k = prob_param[2];
    fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) =
        1. +
        1e-2 *
            std::cos(KG_k * (geomdata.ProbLo(0) + i * geomdata.CellSize(0)) *
                     2. * M_PI /
                     (geomdata.ProbHi(0) -
                      geomdata.ProbLo(
                          0))); // Needs to be 2pi so we have a full period to
                                // satisfy smooth periodic boundary conditions.
    fields(i, j, k, AxKG::getField(AxKG::Fields::KGfv)) = 0.;
    break;
  default:
    amrex::Error("prob_initdata_state: Selected initial condition type is not "
                 "yet implemented!");
  }
}

// initial condition at i,j,k in momentum-space. "state" contains field data (to
// be initialized here), "geomdata" contains the problem geometry, and
// "prob_param" contains the relevant problem parameters.
void AxKG::prob_initdata_mom(
    const int i, const int j, const int k,
    amrex::Array4<amrex::GpuComplex<amrex::Real>> const &fields,
    amrex::GeometryData const &geomdata,
    const amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_param) {
  // For momentum-space ICs we have both real and imaginary parts for each
  // field, so the usual indices from getField() go to 2*getField() for the real
  // component and 2*getField()+1 for the imaginary component.

  static AxKG::ICType ic = AxKG::getIC(prob_param[0]);

  switch (ic) {
  case AxKG::ICType::standard:
    for (int n = 1; n <= 4; n++)
      if (prob_param[n] == PAR_ERR_VAL)
        amrex::Error("prob_initdata_mom: in standard IC---insufficient number "
                     "of problem parameters!");

    // The following needs to be scoped in its own block for the static
    // variables to work
    {
      // Problem variables (and inverse of sqrt(2))
      static amrex::Real KG0 = prob_param[1];
      static amrex::Real A = prob_param[2];
      static amrex::Real B = prob_param[3];
      static amrex::Real cutoff = prob_param[4];
      static amrex::Real invsqrt2 = 1. / std::sqrt(2.);

      // Geometry
      amrex::Real L = geomdata.ProbHi(0) -
                      geomdata.ProbLo(0);    // We assume a cubical geometry.
      amrex::Real dx = geomdata.CellSize(0); // We assume a cubical geometry.
      int N = geomdata.Domain().length(0);   // Assuming a cubical domain

      int ii = i > N / 2 ? N - i : i;
      int jj = j > N / 2 ? N - j : j;
      int kk = k > N / 2 ? N - k : k;

      // Effective mass
      amrex::Real omega =
          (2. * M_PI / L) * (2. * M_PI / L) * (ii * ii + jj * jj + kk * kk) +
          std::abs(Models::compute_model_quantity({1.}, 0, 0., 0., 0.,
                                                  Models::Quant::Vpp));
      if (omega < 0) {
        amrex::Abort("Effective mass is less than 0! Initial conditions must "
                     "not be sufficiently post-inflationary. (Uncomment this "
                     "line to proceed anyway).");
      }
      omega = std::sqrt(std::abs(omega));
      // amrex::Print() << "i = " << i << ", j = " << j << ", k = " << k
      //                << std::endl;
      // amrex::Print() << "--------------------------------------- omega = "
      // << omega << std::endl;

      // Random initializers
      amrex::Real X = rand_uniform();                  // amrex::Random();
      amrex::Real theta1 = 2. * M_PI * rand_uniform(); // amrex::Random();
      amrex::Real theta2 = 2. * M_PI * rand_uniform(); // amrex::Random();

      // LatticeEasy magnitude
      amrex::Real W2 = A * A * B * B * L * L * L /
                       (2. * omega * dx * dx * dx * dx * dx * dx);
      amrex::Real fMag = std::sqrt(-1. * W2 * std::log(X));

      // Return values
      amrex::Real fR, fI, fdR, fdI;

      // Impose a cutoff, if desired
      if (cutoff == 0 || i * i + j * j + k * k < cutoff * cutoff) {
        fR = invsqrt2 * (std::cos(theta1) + std::cos(theta2)) * fMag;
        fI = invsqrt2 * (std::sin(theta1) + std::sin(theta2)) * fMag;
      } else {
        fR = 0;
        fI = 0;
      }

      if (cutoff == 0 || i * i + j * j + k * k < cutoff * cutoff) {
        fdR = (invsqrt2 * (std::sin(theta1) - std::sin(theta2)) * fMag *
               omega); // ori
        fdI = (invsqrt2 * (std::cos(theta2) - std::cos(theta1)) * fMag *
               omega); // ori
#ifdef COMOV_FULL
        fdR += Comoving::get_comoving_ap() * fR * (AxKG::r - 1.);
        fdI += Comoving::get_comoving_ap() * fI * (AxKG::r - 1.);
#endif
      } else {
        fdR = 0;
        fdI = 0;
#ifdef COMOV_FULL
        fdR += Comoving::get_comoving_ap() * fR * (AxKG::r - 1.);
        fdI += Comoving::get_comoving_ap() * fI * (AxKG::r - 1.);
#endif
      }

      fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) =
          amrex::GpuComplex(fR, fI);
      fields(i, j, k, AxKG::getField(AxKG::Fields::KGfv)) =
          amrex::GpuComplex(fdR, fdI);
      // amrex::Print()
      //     << "      fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) "
      //     << fields(i, j, k, AxKG::getField(AxKG::Fields::KGfv)).real()
      //     << std::endl;
    }
    break;

  case AxKG::ICType::delta_k:

    // A simple k-space delta function. The real part of k_x == kG_k, k_y == k_z
    // == 0, is set to KG0, and everything else vanishes.
    if (prob_param[1] == PAR_ERR_VAL || prob_param[2] == PAR_ERR_VAL)
      amrex::Error("prob_initdata_mom: in fixed_k IC---insufficient number of "
                   "problem parameters!");

    // The following needs to be scoped in its own block for the static
    // variables to work
    {
      static amrex::Real KG0 = prob_param[1];
      static amrex::Real KG_k = prob_param[2];

      if (i == KG_k && j == 0 && k == 0)
        fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) =
            amrex::GpuComplex(1., 0.);
      else
        fields(i, j, k, AxKG::getField(AxKG::Fields::KGf)) =
            amrex::GpuComplex(0., 0.);
      fields(i, j, k, AxKG::getField(AxKG::Fields::KGfv)) =
          amrex::GpuComplex(0., 0.);
    }
    break;

  default:
    amrex::Error("prob_initdata_mom: Selected initial condition type is not "
                 "yet implemented!");
  }
}
