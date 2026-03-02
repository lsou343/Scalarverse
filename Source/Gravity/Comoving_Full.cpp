#include <AMReX_BLassert.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <Comoving_Full.H>
#include <atomic>
#include <cmath>

namespace Comoving {
amrex::Real final_a = -1.0;     // If not input always set to -1.0 such that it
                                // doesnt stop at final_a
amrex::Real final_efold = -1.0; // ALternative to final_a
namespace {
amrex::Real A; // Conversion factors for program units
amrex::Real B; // Conversion factors for program units
amrex::Real s;
amrex::Real r;
amrex::Real gridsize; // Grid volume for normalizing energies

amrex::Real a;  // The scale factor at current coarse time
amrex::Real ap; // The derivative of the scale factor at coarse t-0.5dt (or at t
                // for initial time step)
amrex::Real
    app; // The second derivative of the scale factor at current coarse t
amrex::Real a_prev;  // The scale factor at previous coarse time
amrex::Real ap_prev; // The derivative of the scale factor at previous coarse
                     // t-0.5dt (or at t for initial time step)
amrex::Real
    app_prev; // The second derivative of the scale factor at previous coarse t
amrex::Real rho_grad; // Energy density from gradient energies. i.e.: rho_grad =
                      // Sum_Lattice |grad f|^2
amrex::Real rho_pot;  // Energy density from potential energies i.e.: rho_pot  =
                      // Sum_Lattice V(f)
amrex::Real rho_kin;  // Energy density from kinetic energies, just for debug

amrex::Real time = 0.;
amrex::Real prev_time = 0.;

bool init = false; // Have we been initialized?

#ifdef BL_USE_MPI
MPI_Datatype real_type =
    amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type();
#endif

} // Anonymous namespace
// Read stop conditions to stop simualiton at certain efold or scale factor
void read_comov_stop_conditions(amrex::Real a_prev) {
  amrex::ParmParse pp("comov");
  pp.query("final_a", final_a);
  pp.query("final_efold", final_efold);
  if (final_a > 0. && final_efold > 0.) {
    amrex::Abort("Comoving::initComov() Error: Cannot set both final_a and "
                 "final_efold.");
  }
  // calculete final_a from final_efold ( a = a0 * exp(efold) )
  if (final_efold > 0.) {
    final_a = a_prev * std::exp(final_efold);
  }
}
// Init
void initComov(amrex::Real aa, amrex::Real bb, amrex::Real S, amrex::Real R,
               amrex::Real V0, int Gridvol) {
  A = aa;
  B = bb;
  s = S;
  r = R;
  gridsize = Gridvol;
  rho_grad = 0.;
  rho_pot = 0.;
  rho_kin = 0.;

  a = 1.; // We start with a unit scale factor
  // ap = std::sqrt(V0*8.*M_PI/(3.*A*A)); // This is appropriate for \dot{f} = 0
  // initially, not f' = 0.
  ap = std::sqrt(
      8. * M_PI * V0 / (3. * A * A) /
      (1 - 8. * M_PI * r * r /
               (6. * A * A))); // This is for f' = 0 (as in LatticeEasy).

  if (ap < 0. || std::isnan(ap)) {
    // amrex::Print() << "A = " << A << ", B = " << B << ", s = " << s << ", r =
    // " << r << ", V0 = " << V0 << std::endl; amrex::Print() << "M_PI = " <<
    // M_PI << ", ap = " << ap << std::endl;
    amrex::Abort("Comoving::initComov() Error: Initial Hubble value not "
                 "correctly calculated. Check choice of KG0.");
  }

  app = (-s - 2.) * ap * ap + V0 * 8. * M_PI / (A * A);

  a_prev = a;
  ap_prev = ap;
  app_prev = app;

  read_comov_stop_conditions(a_prev);
  init = true;
}

// Reset ap and app once initial field configuration has been set. In principle,
// one could continue the iteration (i.e., use the new ap and app to reset the
// field ICs), but I think this should be good enough.
void set_ics() {

  const amrex::Real coef = (8. * M_PI / 3.) * std::pow(a, -2. * r) / A / A;
  amrex::Real rho = (rho_kin + pow(a, -2. * s - 2.) * rho_grad + rho_pot) /
                    (amrex::Real)gridsize;

  rho *= coef;

  ap = std::sqrt(std::abs(rho));
  app = -(s + 2.) * ap * ap / a +
        (8. * M_PI / A / A) * std::pow(a, -(2. * s + 2. * r + 1.)) *
            (2. * rho_grad / 3. / (amrex::Real)gridsize +
             std::pow(a, 2. * s + 2.) * rho_pot / (amrex::Real)gridsize);
}

// Restart. Note: No need for set_ics() here.
void restartComov(amrex::Real aa, amrex::Real bb, amrex::Real S, amrex::Real R,
                  int Gridvol, amrex::Real Aa, amrex::Real Ap, amrex::Real App,
                  amrex::Real Aa_prev, amrex::Real Ap_prev,
                  amrex::Real App_prev, amrex::Real tt, amrex::Real tt_prev) {
  A = aa;
  B = bb;
  s = S;
  r = R;
  gridsize = Gridvol;
  rho_grad = 0.;
  rho_pot = 0.;
  rho_kin = 0.;

  a = Aa;
  ap = Ap;
  app = App;

  a_prev = Aa_prev;
  ap_prev = Ap_prev;
  app_prev = App_prev;

  time = tt;
  prev_time = tt_prev;

  init = true;

  amrex::ParallelDescriptor::Bcast(
      &a, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::Bcast(
      &ap, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::Bcast(
      &app, 1, amrex::ParallelDescriptor::IOProcessorNumber());

  amrex::ParallelDescriptor::Bcast(
      &a_prev, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::Bcast(
      &ap_prev, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::Bcast(
      &app_prev, 1, amrex::ParallelDescriptor::IOProcessorNumber());

  amrex::ParallelDescriptor::Bcast(
      &time, 1, amrex::ParallelDescriptor::IOProcessorNumber());
  amrex::ParallelDescriptor::Bcast(
      &prev_time, 1, amrex::ParallelDescriptor::IOProcessorNumber());

  // a_prev   = a;
  // ap_prev  = ap;
  // app_prev = app;
}

// Linear interpolation for subcycling
amrex::Real interp(int quant, amrex::Real step) {
  switch (quant) {
  case 0:
    return a_prev + step * (a - a_prev);
  case 1:
    return ap_prev + step * (ap - ap_prev);
  case 2:
    return app_prev + step * (app - app_prev);
  default:
    return -99.;
  }
}

amrex::Real get_comoving_a(amrex::Real tt) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  if (tt == time || tt == 0.)
    return a;
  else if (tt == prev_time)
    return a_prev;
  else {
    amrex::Real step = (tt - prev_time) / (time - prev_time);
    return interp(0, step);
  }
}
amrex::Real get_comoving_ap(amrex::Real tt) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  if (tt == time || tt == 0.)
    return ap;
  else if (tt == prev_time)
    return ap_prev;
  else {
    amrex::Real step = (tt - prev_time) / (time - prev_time);
    return interp(1, step);
  }
}
amrex::Real get_comoving_app(amrex::Real tt) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  if (tt == time || tt == 0.)
    return app;
  else if (tt == prev_time)
    return app_prev;
  else {
    amrex::Real step = (tt - prev_time) / (time - prev_time);
    return interp(2, step);
  }
}

amrex::Real get_rho_g() {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
  return rho_grad;
}
amrex::Real get_rho_v() {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
  return rho_pot;
}
amrex::Real get_rho_t() {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
  return rho_kin;
}

amrex::Real debug_ratio() {

  const amrex::Real coef = (8. * M_PI / 3.) * std::pow(a, -2. * r) / A / A;
  amrex::Real H2 = ap * ap / a / a;
  amrex::Real rho = (rho_kin + pow(a, -2. * s - 2.) * rho_grad + rho_pot) /
                    (amrex::Real)gridsize;
  rho *= coef;

  return H2 / rho;
}

void comoving_est_time_step(amrex::Real &cur_time, amrex::Real &est_dt) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
}

void stop_at_final_a(amrex::Real now_a, amrex::Real final_a) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
  amrex::Abort(
      "Comoving::stop_at_final_a: a_now > a_final, Reached the end condition");
}

void kick_a(amrex::Real dt_half, bool first) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  if (first) {
    a_prev = a;
    ap_prev = ap;
    app_prev = app;
  }

  // Kicks for the scale factor are a little bit more complicated. We are
  // storing ap at t, whereas LatticeEasy stores it at t-1/2dt. That means we
  // have ap available at the right time for the first kick, but we have the
  // wrong ap for the second kick (ap at t has been kicked to t+1/2dt, but for
  // the last kick we need it at t+dt). Therefore, for the second kick we have
  // to use an approximate expression for the acceleration that is used by
  // LatticeEasy.

  //   v_{i+1/2} = v_i + a_i(dt/2)
  //   x_{i+1} = x_i + v_{i+1/2}dt
  //   v_{i+1} = v_{i+1/2} + a_{i+1}(dt/2)

  amrex::Real acc = 0.;

  if (first) // We already calculate app(t) at the end of the second kick, so
             // in principle we should never need to calculate it on the first
             // kick.
  {
    // acc = -(s+2.)*ap*ap/a + (8.*M_PI/A/A)*std::pow(a,
    // -(2.*s+2.*r+1.))*(2.*rho_grad/3./(amrex::Real)gridsize +
    // std::pow(a, 2.*s+2.)*rho_pot/(amrex::Real)gridsize);
    acc = app;
  } else {
    amrex::Real C1 = s + 2., C3 = 2. * s + 2. * r + 2., C4 = 2. * s + 2.;
    acc = -(a / C1 / dt_half) *
          (1. -
           std::sqrt(1. + 4. * dt_half * C1 * ap / a +
                     (4. * dt_half * dt_half * C1 * 8. * M_PI / A / A) *
                         std::pow(a, -C3) *
                         (2. * rho_grad / 3. / (amrex::Real)gridsize +
                          std::pow(a, C4) * rho_pot / (amrex::Real)gridsize)));
    acc -= 2. * ap;
    acc /= 2. * dt_half;
  }

  ap = ap + acc * dt_half;
  if (!first)
    acc = -(s + 2.) * ap * ap / a +
          (8. * M_PI / A / A) * std::pow(a, -(2. * s + 2. * r + 1.)) *
              (2. * rho_grad / 3. / (amrex::Real)gridsize +
               std::pow(a, 2. * s + 2.) * rho_pot / (amrex::Real)gridsize);
  app = acc;
}

void drift_a(amrex::Real dt) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  a = a + ap * dt;

  prev_time = time;
  time += dt;
}

void reset_rho() {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");
  rho_grad = 0.;
  rho_pot = 0.;
  rho_kin = 0.;
}
void add_to_rho(amrex::Real grad, amrex::Real pot, amrex::Real kin) {
  AMREX_ASSERT_WITH_MESSAGE(init, "Must use Comoving::initComov before calling "
                                  "any other Comoving functions.");

  rho_grad += grad;
  rho_pot += pot;
  rho_kin += kin;

#ifdef BL_USE_MPI
  BL_MPI_REQUIRE(MPI_Allreduce(MPI_IN_PLACE, &rho_grad, 1, real_type, MPI_SUM,
                               MPI_COMM_WORLD));
  BL_MPI_REQUIRE(MPI_Allreduce(MPI_IN_PLACE, &rho_pot, 1, real_type, MPI_SUM,
                               MPI_COMM_WORLD));
  BL_MPI_REQUIRE(MPI_Allreduce(MPI_IN_PLACE, &rho_kin, 1, real_type, MPI_SUM,
                               MPI_COMM_WORLD));
#endif
}
amrex::Real get_gridsize() { return gridsize; }
} // namespace Comoving
