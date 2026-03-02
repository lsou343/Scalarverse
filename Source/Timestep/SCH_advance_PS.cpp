#include <AxSCH.H>
#include <AxSCHComov.H>
#ifdef GRAV
#include <AxSP.H>
#include <Gravity.H>
#endif

#include <AMReX_FFT.H>        // for AMReX built-in FFT
#include <AMReX_GpuComplex.H> // for GpuComplex<amrex::Real> or 'Complex'

using namespace amrex;
using Complex = GpuComplex<Real>;

/**
 * We keep the sub-step logic identical to the old code, i.e., "drift" in
 * Fourier space and then "kick" in real space. Now, we do the drift with
 * AMReX's c2c FFT.
 */
namespace {
void FindMaxPsiRealImag(MultiFab &Ax_new) {
  Real max_re_host = -1e30; // Host variable for max real part
  IntVect max_re_idx_host;

  for (MFIter mfi(Ax_new, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrPsi = Ax_new.array(mfi);

    // Allocate a GPU scalar for atomic max tracking
    Gpu::DeviceScalar<Real> max_re_gpu(-1e30);
    Real *max_re_ptr = max_re_gpu.dataPtr();

    // Allocate device storage for index tracking
    Gpu::DeviceVector<int> max_idx_gpu(3, -1);
    int *max_idx_ptr = max_idx_gpu.dataPtr();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Real psi_val = arrPsi(i, j, k);
      Real re =
          arrPsi(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re));

      // Atomic max update
      if (re > *max_re_ptr) {
        amrex::Gpu::Atomic::Max(max_re_ptr, re);
        max_idx_ptr[0] = i;
        max_idx_ptr[1] = j;
        max_idx_ptr[2] = k;
      }
    });

    // Copy results back to host
    Gpu::copy(Gpu::deviceToHost, max_re_ptr, max_re_ptr + 1, &max_re_host);
    Gpu::copy(Gpu::deviceToHost, max_idx_ptr, max_idx_ptr + 3,
              max_re_idx_host.getVect());

    // Synchronize GPU to ensure correctness
    Gpu::streamSynchronize();
  }

  // Print results after computation
  amrex::Print() << "Max Real: " << max_re_host << " at " << max_re_idx_host
                 << "\n";
}
// for debugging:
void FindMaxPsiRealImag(cMultiFab &psi_real) {
  Real max_re_host = -1e30;        // Host variable for max real part
  Real im_if_re_is_max_host = 0.0; // Store corresponding imaginary part
  IntVect max_re_idx_host;

  for (MFIter mfi(psi_real, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrPsi = psi_real.array(mfi);

    // Allocate a GPU scalar for atomic max tracking
    Gpu::DeviceScalar<Real> max_re_gpu(-1e30);
    Real *max_re_ptr = max_re_gpu.dataPtr();

    // Allocate device storage for index and imaginary tracking
    Gpu::DeviceVector<int> max_idx_gpu(3, -1);
    int *max_idx_ptr = max_idx_gpu.dataPtr();
    Gpu::DeviceScalar<Real> im_if_re_is_max_gpu(0.0);
    Real *im_if_re_is_max_ptr = im_if_re_is_max_gpu.dataPtr();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Complex psi_val = arrPsi(i, j, k);
      Real re = psi_val.real();
      Real im = psi_val.imag();

      // Atomic max update
      if (re > *max_re_ptr) {
        amrex::Gpu::Atomic::Max(max_re_ptr, re);
        max_idx_ptr[0] = i;
        max_idx_ptr[1] = j;
        max_idx_ptr[2] = k;
        *im_if_re_is_max_ptr = im;
      }
    });

    // Copy results back to host
    Gpu::copy(Gpu::deviceToHost, max_re_ptr, max_re_ptr + 1, &max_re_host);
    Gpu::copy(Gpu::deviceToHost, max_idx_ptr, max_idx_ptr + 3,
              max_re_idx_host.getVect());
    Gpu::copy(Gpu::deviceToHost, im_if_re_is_max_ptr, im_if_re_is_max_ptr + 1,
              &im_if_re_is_max_host);

    // Synchronize GPU to ensure correctness
    Gpu::streamSynchronize();
  }

  amrex::Print() << "Max value: (" << max_re_host << "," << im_if_re_is_max_host
                 << ") at index " << max_re_idx_host << "\n";
}

// int order = AxSCH::PSorder;
/**
 * \brief Drift step in Fourier space using AMReX's C2C FFT.
 *
 * This replaces the old 'drift' function that relied on hacc::Dfft.
 * We do the forward FFT, multiply by exp(-i*k^2 dt/2m a^2), then inverse FFT,
 * and store the updated wavefunction into psi_real.
 */
static void drift_fft(cMultiFab &psi_real, // in/out wavefunction in real space
                      Real dt, Real h, Real a_half, Real hbaroverm,
                      Geometry const &geom, MultiFab &Ax_new) {
  if (dt == 0.0)
    return;

  // We'll do a forward/backward c2c transform. Typically we can keep the plan
  // static, but for illustration let's create them each time. We transform
  // "psi_real" -> "psi_hat" -> "psi_real".

  FFT::C2C<Real, FFT::Direction::forward> fourier_forward(geom.Domain());
  static bool initialized = false;
  static std::unique_ptr<FFT::C2C<Real, FFT::Direction::forward>> fft_fwd;
  static std::unique_ptr<FFT::C2C<Real, FFT::Direction::backward>> fft_bwd;

  if (!initialized) {
    fft_fwd = std::make_unique<FFT::C2C<Real, FFT::Direction::forward>>(
        geom.Domain());
    fft_bwd = std::make_unique<FFT::C2C<Real, FFT::Direction::backward>>(
        geom.Domain());
    initialized = true;
  }

  // Create cMultiFab for spectral space, consistent with fft_fwd's data layout:
  const auto &[ba_hat, dm_hat] = fft_fwd->getSpectralDataLayout();
  // Ax_new.boxArray(), Ax_new.DistributionMap(), 1, 1
  // cMultiFab psi_hat(ba_hat, dm_hat, 1, 0);
  cMultiFab psi_hat(ba_hat, dm_hat, 1, 0);
  psi_hat.setVal(0.0);
  // 1) Forward FFT: psi_real -> psi_hat
  fft_fwd->forward(psi_real, psi_hat);
  // Print() << "After forward FFT -------psi_hat------\n";
  // FindMaxPsiRealImag(psi_hat);

  Real norm_factor = 1.0_rt / static_cast<Real>(geom.Domain().numPts());
  psi_hat.mult(norm_factor, 0, 1, 0);
  // Print() << "After normalization -------psi_hat------\n";
  // FindMaxPsiRealImag(psi_hat);

  // 2) Multiply each spectral mode by exp(-i * (hbar/m)*k^2/(2*a^2) * dt)
  // const std::complex<double> imagi(0.0, 1.0);
  const Complex imagi(0.0, 1.0);
  // Typically k indices are recognized by i in [0..Nx-1], with shifts for
  // negative frequencies.
  const Real TWO_PI = 2._rt * M_PI;

  const Box domain = geom.Domain();
  int Nx = domain.length(0);
  int Ny = domain.length(1);
  int Nz = domain.length(2);

  const Real dtFactor = -(hbaroverm / (2.0_rt * a_half * a_half)) * dt;
  auto const arrK = psi_hat[0].array();

  for (MFIter mfi(psi_hat); mfi.isValid(); ++mfi) {
    const Box &kb = mfi.validbox();
    auto const arrK = psi_hat.array(mfi);

    ParallelFor(kb, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // "Shift" i,j,k to +/- frequencies
      int ii = (i <= Nx / 2) ? i : (i - Nx);
      int jj = (j <= Ny / 2) ? j : (j - Ny);
      int kk = (k <= Nz / 2) ? k : (k - Nz);

      // wave-vector in each direction
      Real kx = TWO_PI * static_cast<Real>(ii) / static_cast<Real>(Nx);
      Real ky = TWO_PI * static_cast<Real>(jj) / static_cast<Real>(Ny);
      Real kz = TWO_PI * static_cast<Real>(kk) / static_cast<Real>(Nz);

      // dimensionless k^2 scaled by 1/h^2
      Real k2 = (kx * kx + ky * ky + kz * kz) / (h * h);
      // multiply by exp(i * dtFactor * k2)
      Real phase = dtFactor * k2;

      // Complex exp_i_k = std::exp(imagi * phase);
      // std::complex exp_i_k = std::exp(imagi * phase);
      Real exp_i_k_real = std::cos(phase);
      Real exp_i_k_imag = std::sin(phase);

      // arrK(i, j, k) *= Complex(exp_i_k.real(), exp_i_k.imag());
      arrK(i, j, k) *= Complex(exp_i_k_real, exp_i_k_imag);

      // arrK(i, j, k) *= Complex(exp_i_k);
      // arrK(i, j, k) *= 1;
      // if (ii == debug_index && jj == debug_index && kk == debug_index) {
      //   Print() << "k_space array after = " << arrK(i, j, k) <<
      // }
    });
  }
  // Print() << "After drift FFT -------psi_hat------\n";
  // FindMaxPsiRealImag(psi_hat);
  // 4) Normalization: old code does a /N step after each drift in spectral
  // space domain.numPts() = Nx * Ny * Nz

  // 3) Backward FFT: psi_hat -> psi_real
  fft_bwd->backward(psi_hat, psi_real);
  // 4) Normalization: old code does a /N step after each drift in spectral
  // space domain.numPts() = Nx * Ny * Nz
  // Real norm_factor = 1.0_rt / static_cast<Real>(geom.Domain().numPts());
  // psi_real.mult(norm_factor, 0, 1, 0);

  // 5) Update "density" in the real-space cMultiFab if you want it done here.
  //    However, if you want the "density" to live in Ax_new, you'll do it in
  //    the caller or next steps.
  //  // 1.5) Update Density before kick
  for (MFIter mfi(Ax_new, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrA = Ax_new.array(mfi);
    auto const arrPsi = psi_real.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Complex psi_val = arrPsi(i, j, k);
      Real re = psi_val.real();
      Real im = psi_val.imag();

      Real dens = re * re + im * im;
      arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Dens)) = dens;
    });
  };
}

/**
 * \brief Kick step in real space: multiply wavefunction by exp(i * phi *
 * factor).
 *
 * This matches the old code's approach, except we operate on cMultiFab data
 * instead of manual vectors. If dt_d == 0, we skip.
 */
static void kick(cMultiFab &psi_real, MultiFab &phi, Geometry const &geom,
                 Real dt_d, Real a_new, Real a_d, Real hbaroverm) {
  // Recalculate potential if needed (in old code, done outside?)
  // If gravity is defined, user can call gravity->solve_for_new_phi(...)
  // prior to calling 'kick'.
  const std::complex<double> imagi(0.0, 1.0);
  int debug_index = 1;
  const Real factor =
      (a_new / a_d / hbaroverm) * dt_d; // wavefunction multiply factor
  for (MFIter mfi(phi, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrP = phi.array(mfi);
    auto const arrA = psi_real.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real pot = arrP(i, j, k);
      Real phase = pot * factor;
      // Print() << "i, j, k = " << i << ", " << j << ", " << k << std::endl;
      // Print() << "pot = " << pot << std::endl;
      std::complex exp_i = std::exp(imagi * phase);
      arrA(i, j, k) *= Complex(exp_i.real(), exp_i.imag());
      // if (i == debug_index && j == debug_index && k == debug_index) {
      //   Print() << ", factor = " << factor << "\n"
      //           << ", pot = " << pot << "\n"
      //           << ", phase = " << phase << "\n"
      //           << ", exp_i = " << exp_i << "\n"
      //           << ", arrA = " << arrA(i, j, k) << std::endl;
      // }
    });
  }
}

/**
 * \brief Single time-split sub-step: drift (in Fourier) + potential "kick" in
 * real space.
 *
 * This is the new analog to the old "fdm_timestep" that called drift(...) +
 * potential multiplication.
 */
#ifdef GRAV
static void fdm_timestep(cMultiFab &psi_real, MultiFab &phi, Gravity *gravity,
                         Geometry const &geom, int level, Real h, Real dt_c,
                         Real a_c,                        // drift sub-step
                         Real dt_d, Real a_d, Real a_new, // kick sub-step
                         Real hbaroverm, MultiFab &Ax_new) {
  // Print() << "Before drift: " << std::endl;
  // FindMaxPsiRealImag(psi_real);

  // 1) drift
  drift_fft(psi_real, dt_c, h, a_c, hbaroverm, geom, Ax_new);
  psi_real.FillBoundary(geom.periodicity());

  // Print() << "After drift: " << std::endl;
  // FindMaxPsiRealImag(psi_real);
  // 2) optional kick
  if (!dt_d)
    return;
  // Re-calc potential if needed
  if (gravity) {
    // Print() << "gravity is defined" << std::endl;
    // Print() << "phi max before solve--------" << phi.max(0) << std::endl;
    int fill_interior = 0;
    int grav_n_grow = 1;
    gravity->solve_for_new_phi(level, phi, gravity->get_grad_phi_curr(level),
                               fill_interior, grav_n_grow);
    // Print() << "phi max after solve--------" << phi.max(0) << std::endl;
  }
  kick(psi_real, phi, geom, dt_d, a_new, a_d, hbaroverm);
  // Print() << "After kick: " << std::endl;
  // FindMaxPsiRealImag(psi_real);
}
#else
static void fdm_timestep(cMultiFab &psi_real, MultiFab &phi,
                         Geometry const &geom, int level, Real h, Real dt_c,
                         Real a_c, Real dt_d, Real a_d, Real a_new,
                         Real hbaroverm, MultiFab &Ax_new) {
  Print() << "fdm_timestep() is called without gravity.--------------------"
          << std::endl;
  Print() << "Before drift: " << std::endl;
  FindMaxPsiRealImag(psi_real);
  // drift
  drift_fft(psi_real, dt_c, h, a_c, hbaroverm, geom, Ax_new);
  // psi_real.FillBoundary(geom.periodicity());
  Print() << "After drift: " << std::endl;
  FindMaxPsiRealImag(psi_real);

  // optional kick if dt_d != 0
  if (dt_d != 0.0) {
    // (If no gravity, we skip re-calculating potential, but we still do the
    // "kick" if phi is non-zero.)
    kick(psi_real, phi, geom, dt_d, a_new, a_d, hbaroverm);
    Print() << "kick happend" << std::endl;
    Print() << "After kick: " << std::endl;
    FindMaxPsiRealImag(psi_real);
  }
}
#endif
} // end anonymous namespace
/**
 * \brief Main function that replaces "advance_SCH_PS" from old code. Uses
 * AMReX's built-in FFT instead of hacc::Dfft.
 */
void AxSCHComov::advance_SCH_PS(Real time, Real dt, Real a_old, Real a_new) {
  BL_PROFILE("AxSCHComov::advance_FDM_PS()");

  const Real h = geom.CellSize(0);

  // Weight definitions for substepping, identical to old code
  Real w0, w1, w2, w3;
  Real c1, c2, c3, c4, c5, c6, c7, c8;
  Real d1, d2, d3, d4, d5, d6, d7, d8;
  Real a_c1, a_c2, a_c3, a_c4, a_c5, a_c6, a_c7, a_c8;
  Real a_d1, a_d2, a_d3, a_d4, a_d5, a_d6, a_d7, a_d8;
  Real a_time;

  // same code as original to define these weights
  if (AxSCH::PSorder == 6) {
    w1 = -1.17767998417887;
    w2 = 0.235573213359359;
    w3 = 0.784513610477560;
    w0 = 1.0 - 2.0 * (w1 + w2 + w3);

    c1 = (w3 / 2.0) * dt;
    c2 = ((w2 + w3) / 2.0) * dt;
    c3 = ((w1 + w2) / 2.0) * dt;
    c4 = ((w0 + w1) / 2.0) * dt;
    c5 = ((w0 + w1) / 2.0) * dt;
    c6 = ((w1 + w2) / 2.0) * dt;
    c7 = ((w2 + w3) / 2.0) * dt;
    c8 = (w3 / 2.0) * dt;

    d1 = w3 * dt;
    d2 = w2 * dt;
    d3 = w1 * dt;
    d4 = w0 * dt;
    d5 = w1 * dt;
    d6 = w2 * dt;
    d7 = w3 * dt;
    d8 = 0.0;

    a_time = state[getState(StateType::SCH_Type)].prevTime() + 0.5 * c1;
    a_c1 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c1 + c2);
    a_c2 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c2 + c3);
    a_c3 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c3 + c4);
    a_c4 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c4 + c5);
    a_c5 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c5 + c6);
    a_c6 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c6 + c7);
    a_c7 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c7 + c8);
    a_c8 = Comoving::get_comoving_a(a_time);

    a_time = state[getState(StateType::SCH_Type)].prevTime() + 0.5 * d1;
    a_d1 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d1 + d2);
    a_d2 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d2 + d3);
    a_d3 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d3 + d4);
    a_d4 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d4 + d5);
    a_d5 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d5 + d6);
    a_d6 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d6 + d7);
    a_d7 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d7 + d8);
    a_d8 = Comoving::get_comoving_a(a_time);
  } else if (AxSCH::PSorder == 2) {
    c1 = 0.5 * dt;
    c2 = 0.5 * dt;
    d1 = dt;
    d2 = 0.0;

    a_time = state[getState(StateType::SCH_Type)].prevTime() + 0.5 * c1;
    a_c1 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (c1 + c2);
    a_c2 = Comoving::get_comoving_a(a_time);

    a_time = state[getState(StateType::SCH_Type)].prevTime() + 0.5 * d1;
    a_d1 = Comoving::get_comoving_a(a_time);
    a_time += 0.5 * (d1 + d2);
    a_d2 = Comoving::get_comoving_a(a_time);
  } else {
    amrex::Error("Order of algorithm not implemented!");
  }

  // Get old/new wavefunction data
  MultiFab &Ax_old = get_old_data(getState(StateType::SCH_Type));
  MultiFab &Ax_new = get_new_data(getState(StateType::SCH_Type));
  // Print() << "before initializing Ax_old" << '\n';
  // FindMaxPsiRealImag(Ax_old);
  //
  // Print() << "before initializing Ax_new" << '\n';
  // FindMaxPsiRealImag(Ax_new);

  // // Some safety check
  // if (Ax_old.contains_nan(0, Ax_old.nComp(), 0)) {
  //   for (int i = 0; i < Ax_old.nComp(); ++i) {
  //     if (Ax_old.contains_nan(i, 1, 0)) {
  //       amrex::Print() << "NaNs in Ax_old component: " << i << "\n";
  //       Abort("Ax_old has NaNs ::advance_FDM_FFT()");
  //     }
  //   }
  // }

  // Potential
  MultiFab phi(Ax_old.boxArray(), Ax_old.DistributionMap(), 1, 1);
  phi.setVal(0.0);

#ifdef GRAV
  {
    MultiFab &grav_phi =
        get_old_data(AxSP::getState(AxSP::StateType::PhiGrav_Type));
    phi.ParallelCopy(grav_phi, 0, 0, 1, 1, 1,
                     parent->Geom(level).periodicity());
  }
#endif

  // ============== Build cMultiFab from Ax_old for wavefunction in real space
  // ============== We store the wavefunction as 1-component cMultiFab:
  // (Re,Im). Then we copy Ax_old's (SCHf_Re, SCHf_Im) into it.
  cMultiFab psi_real(Ax_old.boxArray(), Ax_old.DistributionMap(), 1,
                     1); // 1 ghost for boundary if you like
  // Print() << "After creaing psi_real: " << std::endl;
  // FindMaxPsiRealImag(psi_real);

  for (MFIter mfi(Ax_old, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    // for (MFIter mfi(Ax_old, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    //   const Box &bx = mfi.tilebox();
    auto const arrOld = Ax_old.array(mfi);
    auto const arrPsi = psi_real.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
      Real re =
          arrOld(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re));
      Real im =
          arrOld(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Im));
      arrPsi(i, j, k) = Complex(re, im);
    });
  }
  // Print() << "After copying Ax_old to psi_real: " << std::endl;
  // FindMaxPsiRealImag(psi_real);

  psi_real.FillBoundary(geom.periodicity());
  // Print() << "After filling boundary psi_real: " << std::endl;
  // FindMaxPsiRealImag(psi_real);

  // ===================== Apply sub-steps (like old code)
  // =====================
#ifdef GRAV
  if (AxSCH::PSorder == 6) {
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c1, a_c1, d1,
                 a_d1, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c2, a_c2, d2,
                 a_d2, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c3, a_c3, d3,
                 a_d3, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c4, a_c4, d4,
                 a_d4, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c5, a_c5, d5,
                 a_d5, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c6, a_c6, d6,
                 a_d6, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c7, a_c7, d7,
                 a_d7, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c8, a_c8, d8,
                 a_d8, a_new, hbaroverm, Ax_new);
  }

  else if (AxSCH::PSorder == 2) {
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c1, a_c1, d1,
                 a_d1, a_new, hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, AxSP::gravity, geom, level, h, c2, a_c2, d2,
                 a_d2, a_new, hbaroverm, Ax_new);
  }
#else
  if (AxSCH::PSorder == 6) {
    fdm_timestep(psi_real, phi, geom, level, h, c1, a_c1, d1, a_d1, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c2, a_c2, d2, a_d2, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c3, a_c3, d3, a_d3, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c4, a_c4, d4, a_d4, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c5, a_c5, d5, a_d5, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c6, a_c6, d6, a_d6, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c7, a_c7, d7, a_d7, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c8, a_c8, d8, a_d8, a_new,
                 hbaroverm, Ax_new);
  }

  else if (AxSCH::PSorder == 2) {
    fdm_timestep(psi_real, phi, geom, level, h, c1, a_c1, d1, a_d1, a_new,
                 hbaroverm, Ax_new);
    fdm_timestep(psi_real, phi, geom, level, h, c2, a_c2, d2, a_d2, a_new,
                 hbaroverm, Ax_new);
  }
#endif

  psi_real.FillBoundary(geom.periodicity());

  // ================== Copy final wavefunction from psi_real -> Ax_new, plus
  // Dens & Phase ===================
  for (MFIter mfi(Ax_new, false); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.validbox();
    auto const arrA = Ax_new.array(mfi);
    auto const arrPsi = psi_real.array(mfi);
    auto const arrOld = Ax_old.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Complex psi_val = arrPsi(i, j, k);
      Real re = psi_val.real();
      Real im = psi_val.imag();

      arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re)) = re;
      arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::SCHf_Im)) = im;

      Real dens = re * re + im * im;
      arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Dens)) = dens;

      Real phase_now = std::atan2(im, re);
      arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Phase)) =
          phase_now;

      // original code does a 2π correction to maintain continuity
      Real old_phase =
          arrOld(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Phase));
      int N = int((old_phase - phase_now) / (2.0 * M_PI));
      if (std::abs(old_phase - phase_now - 2.0 * M_PI * N) < M_PI) {
        arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Phase)) +=
            2.0 * M_PI * N;
      } else {
        // the sign correction from the old code
        arrA(i, j, k, AxSCHComov::getField(AxSCHComov::Fields::Phase)) +=
            2.0 * M_PI * (N + int((0 < old_phase) - (old_phase < 0)));
      }
    });
  }
  Ax_new.FillBoundary(geom.periodicity());

  // // safety check
  // if (Ax_new.contains_nan(0, Ax_new.nComp(), 0)) {
  //   for (int i = 0; i < Ax_new.nComp(); ++i) {
  //     if (Ax_new.contains_nan(i, 1, 0)) {
  //       amrex::Print() << "NaNs in Ax_new component: " << i << "\n";
  //       Abort("Ax_new has NaNs ::advance_FDM_FFT()");
  //     }
  //   }
  // }
  //
  //   Real total_error = 0.0;
  //   Real total_error_real = 0.0;
  //   Real total_error_imag = 0.0;
  //
  //   int grid_count = 0;
  //
  //   // Target indices for specific output (set as needed)
  //   const int target_i = 0; // Adjust to desired i value
  //   const int target_j = 0; // Adjust to desired j value
  //   const int target_k = 0; // Adjust to desired k value
  //
  //   for (MFIter mfi(Ax_old, false); mfi.isValid(); ++mfi) {
  //     const Box &bx = mfi.validbox();
  //     auto const &field_arr = Ax_old.array(mfi);
  //
  //     const Real Lbox = geom.ProbHi(0) - geom.ProbLo(0); // Box length
  //     const Real SCH_k = SCH_k0 * 2. * M_PI / Lbox;
  //     const Real omega = SCH_k * SCH_k * hbaroverm / 2.0; // Dispersion
  //                                                         // relation
  //
  //     ParallelFor(bx, [=, &total_error, &total_error_real, &total_error_imag,
  //                      &grid_count] AMREX_GPU_DEVICE(int i, int j,
  //                                                    int k) noexcept {
  //       // Compute analytical solution
  //       Real x = geom.ProbLo(0) + i * geom.CellSize(0);
  //       Real analytic_re = SCH0 * std::cos(SCH_k * x - omega * time);
  //       Real analytic_im = SCH0 * std::sin(SCH_k * x - omega * time);
  //
  //       // Numerical results
  //       Real numerical_re =
  //           field_arr(i, j, k,
  //           AxSCHComov::getField(AxSCHComov::Fields::SCHf_Re));
  //       Real numerical_im =
  //           field_arr(i, j, k,
  //           AxSCHComov::getField(AxSCHComov::Fields::SCHf_Im));
  //
  //       // Compute errors
  //       Real error_re = std::abs(numerical_re - analytic_re);
  //       Real error_im = std::abs(numerical_im - analytic_im);
  //       Real cell_error = std::sqrt(error_re * error_re + error_im *
  //       error_im);
  //
  //       // Update totals
  //       Gpu::Atomic::Add(&total_error_real, error_re);
  //       Gpu::Atomic::Add(&total_error_imag, error_im);
  //       Gpu::Atomic::Add(&total_error, cell_error);
  //       Gpu::Atomic::Add(&grid_count, 1);
  //
  //       // Print specific grid point data
  //       if (i == target_i && j == target_j && k == target_k) {
  //         Print() << "dx: " << geom.CellSize(0) << ", dt: " << dt << "\n"
  //                 << "i: " << i << ", j: " << j << ", k: " << k << " t = " <<
  //                 time
  //                 << ",  x=" << x << "\n"
  //                 << "Numerical Real: " << numerical_re
  //                 << ", Analytical Real: " << analytic_re << "\n"
  //                 << "Numerical Imag: " << numerical_im
  //                 << ", Analytical Imag: " << analytic_im << "\n";
  //       }
  //     });
  //   }
  //
  //   // Output results
  //   if (ParallelDescriptor::IOProcessor()) {
  //     Print() << "Average Real Error: " << (total_error_real / grid_count)
  //             << "\n";
  //     Print() << "Average Imag Error: " << (total_error_imag / grid_count)
  //             << "\n";
  //     Print() << "------------------------------------------- Average Error:
  //     "
  //             << (total_error / grid_count) << "\n";
  //   }
}
