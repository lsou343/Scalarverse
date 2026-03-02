#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <AMReX_BC_TYPES.H>
#include <AMReX_CONSTANTS.H>
#include <AMReX_Print.H>
#include <AMReX_TagBox.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <BaseAx.H>

#if BL_USE_MPI
#include <MemInfo.H>
#endif

//// Maaaybeee....
#ifdef _OPENMP
#include <omp.h>
#endif

// #include <Prob.H>

// Here we define effectively static variables that are only necessary for the
// functions right here. They are only associated with this compilation unit
// (BaseAx.H/cpp) so don't get passed around unnecessarily.
namespace {

amrex::Real change_max = 1.1;

int slice_int = -1;
std::string slice_file = "slice_";
int slice_nfiles = 128;

int load_balance_int = -1;
int load_balance_wgt_nmax = -1;
int load_balance_strategy = amrex::DistributionMapping::SFC;
// Do we write the particles in single (IEEE32)
//  or double (NATIVE) precision?
#ifdef BL_SINGLE_PRECISION_PARTICLES
std::string particle_plotfile_format = "IEEE32";
#else
std::string particle_plotfile_format = "NATIVE";
#endif

amrex::Real init_shrink = 1.0;
} // namespace

// Class variables that need defining
amrex::Real BaseAx::fixed_dt = -1.0;
amrex::Real BaseAx::initial_dt = -1.0;
amrex::Real BaseAx::dt_cutoff = 0;
int BaseAx::verbose = 0;
int BaseAx::strict_subcycling = 0;
amrex::BCRec BaseAx::phys_bc;
bool BaseAx::dump_old = false;

extern int BaseAx::State_for_Time;

int simd_width = 1;

int BaseAx::nsteps_from_plotfile = -1;

amrex::ErrorList BaseAx::err_list;
amrex::Vector<amrex::AMRErrorTag> BaseAx::errtags;

// int BaseAx::NumSpec  = 0;

// The default for how many digits to use for each column in the runlog
int BaseAx::runlog_precision = 6;

int BaseAx::write_parameters_in_plotfile = 1;
int BaseAx::write_grid_file = 0;
int BaseAx::write_skip_prepost = 0;

// this will be reset upon restart
amrex::Real BaseAx::previousCPUTimeUsed = 0.0;

amrex::Real BaseAx::startCPUTime = 0.0;

extern void prob_errtags_default(amrex::Vector<amrex::AMRErrorTag> &errtags);
///////////////////////////////////////////////////////////////////////
//  The Functions /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////

void BaseAx::variable_cleanup() {

  // For example, in derived classes, you may want to do something like this
  /* if (verbose > 1 && ParallelDescriptor::IOProcessor())
      std::cout << "Deleting gravity in variable_cleanup...\n";
  delete gravity;
  gravity = 0; */

  desc_lst.clear();
}

// Read in the parameters that will be common to any type of run.
void BaseAx::read_params() {
  BL_PROFILE("BaseAx::read_params()");

  amrex::ParmParse pp_ax("ax");

  pp_ax.query("v", verbose);
  pp_ax.query("fixed_dt", fixed_dt);
  pp_ax.query("initial_dt", initial_dt);
  pp_ax.query("change_max", change_max);
  pp_ax.get("dt_cutoff", dt_cutoff);

  // Get boundary conditions
  amrex::Vector<int> lo_bc(BL_SPACEDIM), hi_bc(BL_SPACEDIM);
  pp_ax.getarr("lo_bc", lo_bc, 0, BL_SPACEDIM);
  pp_ax.getarr("hi_bc", hi_bc, 0, BL_SPACEDIM);
  for (int i = 0; i < BL_SPACEDIM; i++) {
    phys_bc.setLo(i, lo_bc[i]);
    phys_bc.setHi(i, hi_bc[i]);
  }

  //
  // Check phys_bc against possible periodic geometry
  // if periodic, must have internal BC marked.
  //
  if (amrex::DefaultGeometry().isAnyPeriodic()) {
    //
    // Do idiot check.  Periodic means interior in those directions.
    //
    for (int dir = 0; dir < BL_SPACEDIM; dir++) {
      if (amrex::DefaultGeometry().isPeriodic(dir)) {
        if (lo_bc[dir] != amrex::PhysBCType::interior) {
          std::cerr << "BaseAx::read_params:periodic in direction " << dir
                    << " but low BC is not Interior" << std::endl;
          amrex::Error();
        }
        if (hi_bc[dir] != amrex::PhysBCType::interior) {
          std::cerr << "BaseAx::read_params:periodic in direction " << dir
                    << " but high BC is not Interior" << std::endl;
          amrex::Error();
        }
      }
    }
  } else {
    //
    // Do idiot check.  If not periodic, should be no interior.
    //
    for (int dir = 0; dir < BL_SPACEDIM; dir++) {
      if (lo_bc[dir] == amrex::PhysBCType::interior) {
        std::cerr << "BaseAx::read_params:interior bc in direction " << dir
                  << " but not periodic" << std::endl;
        amrex::Error();
      }
      if (hi_bc[dir] == amrex::PhysBCType::interior) {
        std::cerr << "BaseAx::read_params:interior bc in direction " << dir
                  << " but not periodic" << std::endl;
        amrex::Error();
      }
    }
  }

  pp_ax.query("strict_subcycling", strict_subcycling);

  pp_ax.query("runlog_precision", runlog_precision);

  pp_ax.query("write_parameter_file", write_parameters_in_plotfile);
  pp_ax.query("write_grid_file", write_grid_file);
  pp_ax.query("write_skip_prepost", write_skip_prepost);

  // How often do we want to write x,y,z 2-d slices of S_new
  pp_ax.query("slice_int", slice_int);
  pp_ax.query("slice_file", slice_file);
  pp_ax.query("slice_nfiles", slice_nfiles);
}

void BaseAx::variable_setup() {
  // initialize the start time for our CPU-time tracker
  startCPUTime = amrex::ParallelDescriptor::second();

  BL_ASSERT(desc_lst.size() == 0);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    const char *amrex_hash = amrex::buildInfoGetGitHash(2);
    std::cout << "\n" << "AMReX git describe: " << amrex_hash << "\n";
  }

  // Get options, set phys_bc
  BaseAx::read_params();

  //
  // DEFINE ERROR ESTIMATION QUANTITIES
  //
  BaseAx::error_setup();
}

BaseAx::BaseAx() {
  BL_PROFILE("BaseAx::BaseAx()");
  fine_mask = 0;
}

BaseAx::BaseAx(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
               const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
               amrex::Real time)
    : amrex::AmrLevel(papa, lev, level_geom, bl, dm, time) {
  BL_PROFILE("BaseAx::BaseAx(Amr)");

  amrex::MultiFab::RegionTag amrlevel_tag("AmrLevel_Level_" +
                                          std::to_string(lev));

  fine_mask = 0;
}

BaseAx::~BaseAx() { delete fine_mask; }

void BaseAx::restart(amrex::Amr &papa, std::istream &is, bool b_read_special) {
  BL_PROFILE("BaseAx::restart()");
  amrex::AmrLevel::restart(papa, is, b_read_special);

  // get the elapsed CPU time to now;
  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    // get elapsed CPU time
    std::ifstream CPUFile;
    std::string FullPathCPUFile = parent->theRestartFile();
    FullPathCPUFile += "/CPUtime";
    CPUFile.open(FullPathCPUFile.c_str(), std::ios::in);

    CPUFile >> previousCPUTimeUsed;
    CPUFile.close();

    std::cout << "read CPU time: " << previousCPUTimeUsed << "\n";
  }
}

void BaseAx::setTimeLevel(amrex::Real time, amrex::Real dt_old,
                          amrex::Real dt_new) {
  if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
    std::cout << "Setting the current time in the state data to "
              << parent->cumTime() << std::endl;
  }
  amrex::AmrLevel::setTimeLevel(time, dt_old, dt_new);
}

void BaseAx::init(amrex::AmrLevel &old) {
  BL_PROFILE("BaseAx::init(old)");

  amrex::Gpu::LaunchSafeGuard lsg(true);
  amrex::MultiFab::RegionTag amrInit_tag("Init_" + std::to_string(level));
  BaseAx *old_level = (BaseAx *)&old;

#ifdef DEBUG
  if (State_for_Time < 0)
    amrex::Abort("Error! Somehow State_for_Time was not reassigned in the "
                 "derived class!");
#endif
  //
  // Create new grid data by fillpatching from old.
  //
  amrex::Real dt_new = parent->dtLevel(level);

  amrex::Real cur_time = old_level->state[State_for_Time].curTime();
  amrex::Real prev_time = old_level->state[State_for_Time].prevTime();

  amrex::Real dt_old = cur_time - prev_time;
  setTimeLevel(cur_time, dt_old, dt_new);
}

//
// This version inits the data on a new level that did not
// exist before regridding.
//
void BaseAx::init() {
  BL_PROFILE("BaseAx::init()");
  amrex::Real dt = parent->dtLevel(level);

  amrex::Real cur_time = get_level(level - 1).state[State_for_Time].curTime();
  amrex::Real prev_time = get_level(level - 1).state[State_for_Time].prevTime();

  amrex::Real dt_old =
      (cur_time - prev_time) / (amrex::Real)parent->MaxRefRatio(level - 1);

  setTimeLevel(cur_time, dt_old, dt);
}

amrex::Real BaseAx::initial_time_step() {
  BL_PROFILE("BaseAx::initial_time_step()");
  amrex::Real dummy_dt = 0;
  amrex::Real init_dt = 0;

  if (initial_dt > 0) {
    init_dt = initial_dt;
  } else {
    init_dt = init_shrink * est_time_step(dummy_dt);
  }

  return init_dt;
}

void BaseAx::computeNewDt(int finest_level, int sub_cycle,
                          amrex::Vector<int> &n_cycle,
                          const amrex::Vector<amrex::IntVect> &ref_ratio,
                          amrex::Vector<amrex::Real> &dt_min,
                          amrex::Vector<amrex::Real> &dt_level,
                          amrex::Real stop_time, int post_regrid_flag) {
  BL_PROFILE("BaseAx::computeNewDt()");
  //
  // We are at the start of a coarse grid timecycle.
  // Compute the timesteps for the next iteration.
  //
  if (level > 0)
    return;

  int i;

  amrex::Real dt_0 = 1.0e+100;
  int n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    BaseAx &adv_level = get_level(i);
    dt_min[i] = adv_level.est_time_step(dt_level[i]);
  }

  if (fixed_dt <= 0.0) {
    if (post_regrid_flag == 1) {
      //
      // Limit dt's by pre-regrid dt
      //
      for (i = 0; i <= finest_level; i++) {
        dt_min[i] = std::min(dt_min[i], dt_level[i]);
      }
      //
      // Find the minimum over all levels
      //
      for (i = 0; i <= finest_level; i++) {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0, n_factor * dt_min[i]);
      }
    } else {
      bool sub_unchanged = true;
      if ((parent->maxLevel() > 0) && (level == 0) &&
          (parent->subcyclingMode() == "Optimal") &&
          (parent->okToRegrid(level) || parent->levelSteps(0) == 0)) {
        int new_cycle[finest_level + 1];
        for (i = 0; i <= finest_level; i++)
          new_cycle[i] = n_cycle[i];
        // The max allowable dt
        amrex::Real dt_max[finest_level + 1];
        for (i = 0; i <= finest_level; i++) {
          dt_max[i] = dt_min[i];
        }
        // find the maximum number of cycles allowed.
        int cycle_max[finest_level + 1];
        cycle_max[0] = 1;
        for (i = 1; i <= finest_level; i++) {
          cycle_max[i] = parent->MaxRefRatio(i - 1);
        }
        // estimate the amout of work to advance each level.
        amrex::Real est_work[finest_level + 1];
        for (i = 0; i <= finest_level; i++) {
          est_work[i] = parent->getLevel(i).estimateWork();
        }
        // this value will be used only if the subcycling pattern is changed.
        dt_0 = parent->computeOptimalSubcycling(finest_level + 1, new_cycle,
                                                dt_max, est_work, cycle_max);
        for (i = 0; i <= finest_level; i++) {
          if (n_cycle[i] != new_cycle[i]) {
            sub_unchanged = false;
            n_cycle[i] = new_cycle[i];
          }
        }
      }

      if (sub_unchanged)
      //
      // Limit dt's by change_max * old dt
      //
      {
        for (i = 0; i <= finest_level; i++) {
          if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
            if (dt_min[i] > change_max * dt_level[i]) {
              std::cout << "BaseAx::compute_new_dt : limiting dt at level " << i
                        << '\n';
              std::cout << " ... new dt computed: " << dt_min[i] << '\n';
              std::cout << " ... but limiting to: " << change_max * dt_level[i]
                        << " = " << change_max << " * " << dt_level[i] << '\n';
            }
          }

          dt_min[i] = std::min(dt_min[i], change_max * dt_level[i]);
        }
        //
        // Find the minimum over all levels
        //
        for (i = 0; i <= finest_level; i++) {
          n_factor *= n_cycle[i];
          dt_0 = std::min(dt_0, n_factor * dt_min[i]);
        }
      } else {
        if (verbose && amrex::ParallelDescriptor::IOProcessor()) {
          std::cout << "BaseAx: Changing subcycling pattern. New pattern:\n";
          for (i = 1; i <= finest_level; i++)
            std::cout << "   Lev / n_cycle: " << i << " " << n_cycle[i] << '\n';
        }
      }
    }
  } else {
    dt_0 = fixed_dt;
  }

  //
  // Limit dt's by the value of stop_time.
  //
  const amrex::Real eps = 0.001 * dt_0;
  amrex::Real cur_time = state[State_for_Time].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / n_factor;
  }
}

void BaseAx::computeInitialDt(int finest_level, int sub_cycle,
                              amrex::Vector<int> &n_cycle,
                              const amrex::Vector<amrex::IntVect> &ref_ratio,
                              amrex::Vector<amrex::Real> &dt_level,
                              amrex::Real stop_time) {
  BL_PROFILE("BaseAx::computeInitialDt()");
  //
  // Grids have been constructed, compute dt for all levels.
  //
  if (level > 0)
    return;

  int i;
  amrex::Real dt_0 = 1.0e+100;
  int n_factor = 1;
  if (parent->subcyclingMode() == "Optimal") {
    int new_cycle[finest_level + 1];
    for (i = 0; i <= finest_level; i++)
      new_cycle[i] = n_cycle[i];
    amrex::Real dt_max[finest_level + 1];
    for (i = 0; i <= finest_level; i++) {
      dt_max[i] = get_level(i).initial_time_step();
    }
    // Find the maximum number of cycles allowed
    int cycle_max[finest_level + 1];
    cycle_max[0] = 1;
    for (i = 1; i <= finest_level; i++) {
      cycle_max[i] = parent->MaxRefRatio(i - 1);
    }
    // estimate the amout of work to advance each level.
    amrex::Real est_work[finest_level + 1];
    for (i = 0; i <= finest_level; i++) {
      est_work[i] = parent->getLevel(i).estimateWork();
    }
    dt_0 = parent->computeOptimalSubcycling(finest_level + 1, new_cycle, dt_max,
                                            est_work, cycle_max);
    for (i = 0; i <= finest_level; i++) {
      n_cycle[i] = new_cycle[i];
    }
    if (verbose && amrex::ParallelDescriptor::IOProcessor() &&
        finest_level > 0) {
      std::cout << "BaseAx: Initial subcycling pattern:\n";
      for (i = 0; i <= finest_level; i++)
        std::cout << "Level " << i << ": " << n_cycle[i] << '\n';
    }
  } else {
    for (i = 0; i <= finest_level; i++) {
      dt_level[i] = get_level(i).initial_time_step();
      n_factor *= n_cycle[i];
      dt_0 = std::min(dt_0, n_factor * dt_level[i]);
    }
  }
  //
  // Limit dt's by the value of stop_time.
  //
  const amrex::Real eps = 0.001 * dt_0;
  amrex::Real cur_time = state[State_for_Time].curTime();
  if (stop_time >= 0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  n_factor = 1;
  for (i = 0; i <= finest_level; i++) {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / n_factor;
  }
}

// Implement if you think of something that needs to be done at the end of each
// time step.
/* void BaseAx::post_timestep (int iteration)
{
} */

void BaseAx::average_down(int state_index) {
  BL_PROFILE("BaseAx::average_down(si)");

  if (level == parent->finestLevel())
    return;

  BaseAx &fine_lev = get_level(level + 1);

  const amrex::Geometry &fgeom = fine_lev.geom;
  const amrex::Geometry &cgeom = geom;

  amrex::MultiFab &S_crse = get_new_data(state_index);
  amrex::MultiFab &S_fine = fine_lev.get_new_data(state_index);

  const int num_comps = S_fine.nComp();

  amrex::average_down(S_fine, S_crse, fgeom, cgeom, 0, num_comps, fine_ratio);
}

// Nothing to do in general. --PH
void BaseAx::post_restart() {}

void BaseAx::postCoarseTimeStep(amrex::Real cumtime) {
  BL_PROFILE("BaseAx::postCoarseTimeStep()");
  amrex::MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

  if (load_balance_int >= 0 && nStep() % load_balance_int == 0) {
    if (verbose > 0)
      amrex::Print() << "Load balancing since " << nStep() << " mod "
                     << load_balance_int << " == 0" << std::endl;

    for (int lev = 0; lev <= parent->finestLevel(); lev++) {

      amrex::Vector<long> wgts(grids.size());
      amrex::DistributionMapping dm;

      for (unsigned int i = 0; i < wgts.size(); i++) {
        wgts[i] = grids[i].numPts();
      }
      if (load_balance_strategy ==
          amrex::DistributionMapping::Strategy::KNAPSACK)
        dm.KnapSackProcessorMap(wgts, load_balance_wgt_nmax);
      else if (load_balance_strategy ==
               amrex::DistributionMapping::Strategy::SFC)
        dm.SFCProcessorMap(grids, wgts, load_balance_wgt_nmax);
      else if (load_balance_strategy ==
               amrex::DistributionMapping::Strategy::ROUNDROBIN)
        dm.RoundRobinProcessorMap(wgts, load_balance_wgt_nmax);

      amrex::Gpu::streamSynchronize();
    }
  }

  amrex::AmrLevel::postCoarseTimeStep(cumtime);

  if (verbose > 1) {
    amrex::Print() << "End of postCoarseTimeStep, printing:" << std::endl;
    amrex::MultiFab::printMemUsage();
    amrex::Arena::PrintUsage();
  }
}

void BaseAx::post_regrid(int lbase, int new_finest) {
  BL_PROFILE("BaseAx::post_regrid()");

  delete fine_mask;
  fine_mask = 0;
}

void BaseAx::post_init(amrex::Real stop_time) {
  BL_PROFILE("BaseAx::post_init()");
  if (level > 0) {
    return;
  }

  // If we restarted from a plotfile, we need to reset the level_steps counter
  if (!parent->theRestartPlotFile().empty()) {
    parent->setLevelSteps(0, nsteps_from_plotfile);
  }

  amrex::Gpu::LaunchSafeGuard lsg(true);
  //
  // Average data down from finer levels
  // so that conserved data is consistent between levels.
  //
  int finest_level = parent->finestLevel();
  for (int k = finest_level - 1; k >= 0; --k) {
    get_level(k).average_down();
  }

  write_info();
}

int BaseAx::okToContinue() {
  if (level > 0) {
    return 1;
  }

  int test = 1;
  if (parent->dtLevel(0) < dt_cutoff) {
    test = 0;
  }

  return test;
}

void BaseAx::errorEst(amrex::TagBoxArray &tags, int clearval, int tagval,
                      amrex::Real time, int n_error_buf, int ngrow) {
  BL_PROFILE("BaseAx::errorEst()");

  amrex::Print() << "ErrTags Debug Here \n";
  for (int j = 0; j < errtags.size(); ++j) {
    amrex::Print() << errtags[j].Field();
  }

  for (int j = 0; j < errtags.size(); ++j) {
    std::unique_ptr<amrex::MultiFab> mf;
    if (errtags[0].Field() != std::string()) {
      mf = std::unique_ptr<amrex::MultiFab>(
          derive(errtags[j].Field(), time, errtags[j].NGrow()));
    }
    errtags[j](tags, mf.get(), clearval, tagval, time, level, geom);
  }
}

std::unique_ptr<amrex::MultiFab> BaseAx::derive(const std::string &name,
                                                amrex::Real time, int ngrow) {
  BL_PROFILE("BaseAx::derive()");

  // amrex::Gpu::LaunchSafeGuard lsg(true);

  //     return particle_derive(name, time, ngrow);  // particle_derive is a Nyx
  //     function that handles derived quantities for particles
  //                                                 // AND handles the default
  //                                                 case if there are no
  //                                                 particles.

  return AmrLevel::derive(name, time, ngrow);
}

void BaseAx::derive(const std::string &name, amrex::Real time,
                    amrex::MultiFab &mf, int dcomp) {
  BL_PROFILE("BaseAx::derive(mf)");

  //     const auto& derive_dat = particle_derive(name, time, mf.nGrow());  //
  //     particle_derive is a Nyx function that handles derived quantities for
  //     particles
  // //                                                                     //
  // AND handles the default case if there are no particles.

  //     MultiFab::Copy(mf, *derive_dat, 0, dcomp, 1, mf.nGrow());
  // if (name == "Rank")
  // {
  const auto &derive_dat = derive(name, time, mf.nGrow());
  amrex::MultiFab::Copy(mf, *derive_dat, 0, dcomp, 1, mf.nGrow());
  // }
}

amrex::Real BaseAx::getCPUTime() {

  int numCores = amrex::ParallelDescriptor::NProcs();
#ifdef _OPENMP
  numCores = numCores * omp_get_max_threads();
#endif

  amrex::Real T =
      numCores * (amrex::ParallelDescriptor::second() - startCPUTime) +
      previousCPUTimeUsed;

  return T;
}

amrex::MultiFab *BaseAx::build_fine_mask() {
  BL_ASSERT(level > 0); // because we are building a mask for the coarser level

  if (fine_mask != 0)
    return fine_mask;

  amrex::BoxArray baf = parent->boxArray(level);
  baf.coarsen(crse_ratio); // N.B.: crse_ratio is a member of AmrLevel.

  const amrex::BoxArray &bac = parent->boxArray(level - 1);
  fine_mask =
      new amrex::MultiFab(bac, parent->DistributionMap(level - 1), 1, 0);
  fine_mask->setVal(1.0);

#ifdef _OPENMP
#pragma omp parallel
#endif
  for (amrex::MFIter mfi(*fine_mask); mfi.isValid(); ++mfi) {
    amrex::FArrayBox &fab = (*fine_mask)[mfi];

    std::vector<std::pair<int, amrex::Box>> isects =
        baf.intersections(fab.box());

    for (int ii = 0; ii < isects.size(); ii++) {
      fab.setVal<amrex::RunOn::Host>(0.0, isects[ii].second, 0);
    }
  }

  return fine_mask;
}

void BaseAx::LevelDirectoryNames(const std::string &dir,
                                 const std::string &secondDir,
                                 std::string &LevelDir, std::string &FullPath) {
  LevelDir = amrex::Concatenate("Level_", level, 1);
  //
  // Now for the full pathname of that directory.
  //
  FullPath = dir;
  if (!FullPath.empty() && FullPath.back() != '/') {
    FullPath += '/';
  }
  FullPath += secondDir;
  FullPath += "/";
  FullPath += LevelDir;
}

void BaseAx::CreateLevelDirectory(const std::string &dir) {
  amrex::AmrLevel::CreateLevelDirectory(
      dir); // ---- this sets levelDirectoryCreated = true

  // Can do more here in derived classes, particularly those with particles.
}

void BaseAx::error_setup() {
  std::string amr_prefix = "amr";
  amrex::ParmParse ppamr(amr_prefix);
  amrex::Vector<std::string> refinement_indicators;
  ppamr.queryarr("refinement_indicators", refinement_indicators, 0,
                 ppamr.countval("refinement_indicators"));

  // I don't see any reason to have default error tags, just specify what you
  // want to refine in the inputs file. if (refinement_indicators.size()==0)
  //     prob_errtags_default(errtags);
  //     // f(errtags);

  // else {
  for (int i = 0; i < refinement_indicators.size(); ++i) {
    std::string ref_prefix = amr_prefix + "." + refinement_indicators[i];
    amrex::ParmParse ppr(ref_prefix);
    amrex::RealBox realbox;
    if (ppr.countval("in_box_lo")) {
      std::vector<amrex::Real> box_lo(BL_SPACEDIM), box_hi(BL_SPACEDIM);
      ppr.getarr("in_box_lo", box_lo, 0, box_lo.size());
      ppr.getarr("in_box_hi", box_hi, 0, box_hi.size());
      realbox = amrex::RealBox(&(box_lo[0]), &(box_hi[0]));
    }
    amrex::AMRErrorTagInfo info;
    if (realbox.ok()) {
      info.SetRealBox(realbox);
    }
    if (ppr.countval("start_time") > 0) {
      amrex::Real min_time;
      ppr.get("start_time", min_time);
      info.SetMinTime(min_time);
    }
    if (ppr.countval("end_time") > 0) {
      amrex::Real max_time;
      ppr.get("end_time", max_time);
      info.SetMaxTime(max_time);
    }
    if (ppr.countval("max_level") > 0) {
      int max_level;
      ppr.get("max_level", max_level);
      if (max_level >= 0)
        info.SetMaxLevel(max_level);
      else
        info.SetMaxLevel(0);
    }
    if (ppr.countval("value_greater")) {
      amrex::Vector<amrex::Real> value;
      int nlevs = ppr.countval("value_greater");
      value.resize(nlevs);
      ppr.getarr("value_greater", value, 0, nlevs);
      // Real value; ppr.get("value_greater",value);
      std::string field;
      ppr.get("field_name", field);
      errtags.push_back(
          amrex::AMRErrorTag(value, amrex::AMRErrorTag::GREATER, field, info));
    } else if (ppr.countval("value_less")) {
      amrex::Real value;
      ppr.get("value_less", value);
      std::string field;
      ppr.get("field_name", field);
      errtags.push_back(
          amrex::AMRErrorTag(value, amrex::AMRErrorTag::LESS, field, info));
    } else if (ppr.countval("vorticity_greater")) {
      amrex::Real value;
      ppr.get("vorticity_greater", value);
      const std::string field = "mag_vort";
      errtags.push_back(
          amrex::AMRErrorTag(value, amrex::AMRErrorTag::VORT, field, info));
    } else if (ppr.countval("adjacent_difference_greater")) {
      amrex::Real value;
      ppr.get("adjacent_difference_greater", value);
      std::string field;
      ppr.get("field_name", field);
      errtags.push_back(
          amrex::AMRErrorTag(value, amrex::AMRErrorTag::GRAD, field, info));
    } else if (realbox.ok()) {
      errtags.push_back(amrex::AMRErrorTag(info));
    } else {
      amrex::Abort(std::string("Unrecognized refinement indicator for " +
                               refinement_indicators[i])
                       .c_str());
    }
  }
  // }
}

void BaseAx::manual_tags_placement(
    amrex::TagBoxArray &tags, const amrex::Vector<amrex::IntVect> &bf_lev) {}

void BaseAx::checkPoint(const std::string &dir, std::ostream &os,
                        amrex::VisMF::How how, bool dump_old_default) {

  for (int s = 0; s < desc_lst.size(); ++s) {
    if (dump_old && state[s].hasOldData()) {
      amrex::MultiFab &old_MF = get_old_data(s);
      amrex::prefetchToHost(old_MF);
    }
    amrex::MultiFab &new_MF = get_new_data(s);
    amrex::prefetchToHost(new_MF);
  }

  amrex::AmrLevel::checkPoint(dir, os, how, dump_old);

  for (int s = 0; s < desc_lst.size(); ++s) {
    if (dump_old && state[s].hasOldData()) {
      amrex::MultiFab &old_MF = get_old_data(s);
      amrex::prefetchToDevice(old_MF);
    }
    amrex::MultiFab &new_MF = get_new_data(s);
    amrex::prefetchToDevice(new_MF);
  }

  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    writeJobInfo(dir);
  }

  if (level == 0 && amrex::ParallelDescriptor::IOProcessor()) {
    {
      // store elapsed CPU time
      std::ofstream CPUFile;
      std::string FullPathCPUFile = dir;
      FullPathCPUFile += "/CPUtime";
      CPUFile.open(FullPathCPUFile.c_str(), std::ios::out);

      CPUFile << std::setprecision(15) << getCPUTime();
      CPUFile.close();
    }
  }
}

void BaseAx::post_timestep(int iteration) {
  // Lots to do here if we have particles.

  BL_PROFILE("BaseAx::post_timestep()");

  amrex::Gpu::LaunchSafeGuard lsg(true);
  amrex::MultiFab::RegionTag amrPost_tag("Post_" + std::to_string(level));

  //
  // Integration cycle on fine level grids is complete
  // do post_timestep stuff here.
  //
  int finest_level = parent->finestLevel();
  const int ncycle = parent->nCycle(level);

  amrex::Gpu::streamSynchronize();

  if (level < finest_level)
    this->average_down();

  amrex::Gpu::streamSynchronize();
  BL_PROFILE_VAR("BaseAx::post_timestep()::sum_write", sum_write);

  if (level == 0) {
    int nstep = parent->levelSteps(0);

    write_info();

#if BL_USE_MPI
    // Memory monitoring:
    MemInfo *mInfo = MemInfo::GetInstance();
    char info[32];
    snprintf(info, sizeof(info), "Step %4d", nstep);
    mInfo->LogSummary(info);
#endif
  }

  amrex::Gpu::streamSynchronize();
  BL_PROFILE_VAR_STOP(sum_write);
}

std::string BaseAx::thePlotFileType() const {
  //
  // Increment this whenever the writePlotFile() format changes.
  //
  static const std::string the_plot_file_type("HyperCLaw-V1.1");
  return the_plot_file_type;
}

// void BaseAx::prob_initdata_mom_on_mf(amrex::MultiFab &mf ,
//                                 amrex::Geometry const& geom,
//                                 const
//                                 amrex::GpuArray<amrex::Real,BaseAx::max_prob_param>&
//                                 prob_param)
// {
//     BL_PROFILE("prob_initdata_mom_on_mf")

//     amrex::FFT::R2C fft(geom.Domain());
//     auto const& [ba,dm] = fft.getSpectralDataLayout();

//     amrex::cMultiFab fillK(ba, dm, mf.nComp(), 0);

//     // Initialize the Fourier data
//     for (amrex::MFIter mfi(fillK,amrex::TilingIfNotGPU()); mfi.isValid();
//     ++mfi)
//     {
//         const amrex::Box& bx  = mfi.tilebox();
//         const auto fab_new = fillK.array(mfi);

//         amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k)
//         noexcept
//         // NOTE THE & instead of =!! This takes references for external
//         variables as opposed to making local copies.
//         {
//             prob_initdata_mom(i, j ,k, fab_new, geom.data(), prob_param);

//             // Set the 0-mode to 0 since we add it in later
//             if(i == 0 && j == 0 && k == 0)
//             {
//                 for(int comp = 0; comp < fillK.nComp(); comp++)
//                 fab_new(0, 0, 0, comp) = amrex::GpuComplex(0.,0.);  // Set
//                 the 0-mode to 0 since we add it in later
//             }
//         });
//     }

//     // amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::backward>
//     fft_backward(geom.Domain());
//     // fft_backward.backward(fillK, mf);
//     fft.backward(fillK, mf);
//     mf.mult(1./mf.boxArray().numPts());
//     mf.FillBoundary(geom.periodicity());
// }

// A tool to spit out a plot file at a given interval or event. In Nyx, this was
// used to specify a list of redshifts to output plot files at. Not needed here,
// and there is a default implementation in AmrLevel, `return false`.  --PH
/* bool BaseAx::writePlotNow ()
{
    BL_PROFILE("BaseAx::writePlotNow()");
    if (level > 0)
        amrex::Error("Should only call writePlotNow at level 0!");

    bool found_one = false;

    if (plot_z_values.size() > 0)
    {
        Real prev_time = state[State_for_Time].prevTime();
        Real  cur_time = state[State_for_Time].curTime();

        Real a_old = get_comoving_a(prev_time);
        Real z_old = (1. / a_old) - 1.;

        Real a_new = get_comoving_a( cur_time);
        Real z_new = (1. / a_new) - 1.;

        for (int i = 0; i < plot_z_values.size(); i++)
        {
            if (std::abs(z_new - plot_z_values[i]) < (0.01 * (z_old - z_new)) )
                found_one = true;
        }
    }

    if (found_one) {
        return true;
    } else {
        return false;
    }
} */

// Same story as writePlotNow
/* bool Nyx::doAnalysisNow ()
{
    BL_PROFILE("Nyx::doAnalysisNow()");
    if (level > 0)
        amrex::Error("Should only call doAnalysisNow at level 0!");

    bool found_one = false;

    if (analysis_z_values.size() > 0)
    {

        Real prev_time = state[State_for_Time].prevTime();
        Real  cur_time = state[State_for_Time].curTime();

        Real a_old = get_comoving_a(prev_time);
        Real z_old = (1. / a_old) - 1.;

        Real a_new = get_comoving_a( cur_time);
        Real z_new = (1. / a_new) - 1.;

        for (int i = 0; i < analysis_z_values.size(); i++)
        {
            if (std::abs(z_new - analysis_z_values[i]) < (0.01 * (z_old -
z_new)) ) found_one = true;
        }
    }

    if (found_one) {
        return true;
    } else {
        return false;
    }
} */

/* void BaseAx::writeMultiFabAsPlotFile(const std::string& pltfile,
                             const amrex::MultiFab&    mf,
                             std::string        componentName)
{
    std::ofstream os;
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if( ! amrex::UtilCreateDirectory(pltfile, 0755)) {
          amrex::CreateDirectoryFailed(pltfile);
        }
        std::string HeaderFileName = pltfile + "/Header";
        os.open(HeaderFileName.c_str(),
std::ios::out|std::ios::trunc|std::ios::binary);
        // The first thing we write out is the plotfile type.
        os << thePlotFileType() << '\n';
        // Just one component ...
        os << 1 << '\n';
        // ... with name
        os << componentName << '\n';
        // Dimension
        os << BL_SPACEDIM << '\n';
        // Time
        os << "0\n";
        // One level
        os << "0\n";
        for (int i = 0; i < BL_SPACEDIM; i++)
            os << Geom().ProbLo(i) << ' ';
        os << '\n';
        for (int i = 0; i < BL_SPACEDIM; i++)
            os << Geom().ProbHi(i) << ' ';
        os << '\n';
        // Only one level -> no refinement ratios
        os << '\n';
        // Geom
        os << parent->Geom(0).Domain() << ' ';
        os << '\n';
        os << parent->levelSteps(0) << ' ';
        os << '\n';
        for (int k = 0; k < BL_SPACEDIM; k++)
            os << parent->Geom(0).CellSize()[k] << ' ';
        os << '\n';
        os << (int) Geom().Coord() << '\n';
        os << "0\n"; // Write bndry data.
    }
    // Build the directory to hold the MultiFab at this level.
    // The name is relative to the directory containing the Header file.
    //
    static const std::string BaseName = "/Cell";
    std::string Level = "Level_0";
    //
    // Now for the full pathname of that directory.
    //
    std::string FullPath = pltfile;
    if ( ! FullPath.empty() && FullPath[FullPath.size()-1] != '/') {
        FullPath += '/';
    }
    FullPath += Level;
    //
    // Only the I/O processor makes the directory if it doesn't already exist.
    //
    if (ParallelDescriptor::IOProcessor()) {
        if ( ! amrex::UtilCreateDirectory(FullPath, 0755)) {
            amrex::CreateDirectoryFailed(FullPath);
        }
    }
    //
    // Force other processors to wait until directory is built.
    //
    amrex::ParallelDescriptor::Barrier();

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        amrex::Real cur_time = state[State_for_Time].curTime();
        os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
        os << parent->levelSteps(level) << '\n';

        for (int i = 0; i < grids.size(); ++i)
        {
            amrex::RealBox gridloc = amrex::RealBox(grids[i], geom.CellSize(),
geom.ProbLo()); for (int n = 0; n < BL_SPACEDIM; n++) os << gridloc.lo(n) << ' '
<< gridloc.hi(n) << '\n';
        }
        //
        // The full relative pathname of the MultiFabs at this level.
        // The name is relative to the Header file containing this name.
        // It's the name that gets written into the Header.
        //
        std::string PathNameInHeader = Level;
        PathNameInHeader += BaseName;
        os << PathNameInHeader << '\n';
    }

    //
    // Use the Full pathname when naming the MultiFab.
    //
    std::string TheFullPath = FullPath;
    TheFullPath += BaseName;
    amrex::VisMF::Write(mf, TheFullPath);
    amrex::ParallelDescriptor::Barrier();
} */
