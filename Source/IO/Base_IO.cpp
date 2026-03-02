#include <BaseAx.H>

#include <unistd.h>

extern std::string inputs_name;

void BaseAx::writePlotFilePost(const std::string& dir, std::ostream& os)
{

    amrex::Gpu::LaunchSafeGuard lsg(true);
    amrex::Real cur_time = state[State_for_Time].curTime(); 

    if(write_skip_prepost == 1)
    {
        amrex::Print() << "Skip writePlotFilePost" << std::endl;
    }
    else
    {


        if (level == 0 && amrex::ParallelDescriptor::IOProcessor())
        {
            writeJobInfo(dir);
        }

        // Write out all parameters into the plotfile
        if (write_parameters_in_plotfile) 
        {
            write_parameter_file(dir);
        }

        if (write_grid_file)
        {
            std::string FullPathGrid = dir;
            FullPathGrid += "/grids_file";
            print_grids(FullPathGrid);
        }

        /* if(Nyx::theDMPC()) {
        Nyx::theDMPC()->SetLevelDirectoriesCreated(false);
        } */


        /* if(Nyx::theDMPC()) {
        Nyx::theDMPC()->WritePlotFilePost();
        } */
    }
    if(verbose) 
    {

        if (level == 0)
        {
            amrex::Print().SetPrecision(15) << "Output file " << dir << " at time " << std::to_string(cur_time) << " and step " << std::to_string(nStep()) << std::endl;
        }
    }

}

void BaseAx::writeJobInfo (const std::string& dir)
{
        // job_info file with details about the run
        std::ofstream jobInfoFile;
        std::string FullPathJobInfoFile = dir;
        FullPathJobInfoFile += "/job_info";
        jobInfoFile.open(FullPathJobInfoFile.c_str(), std::ios::out);

        std::string PrettyLine = std::string(78, '=') + "\n";
        std::string OtherLine = std::string(78, '-') + "\n";
        std::string SkipSpace = std::string(8, ' ');

        // job information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Nyx Job Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "inputs file: " << inputs_name << "\n\n";

        jobInfoFile << "number of MPI processes: " << amrex::ParallelDescriptor::NProcs() << "\n";
#ifdef _OPENMP
        jobInfoFile << "number of threads:       " << omp_get_max_threads() << "\n";
#endif
        jobInfoFile << "\n";
        jobInfoFile << "CPU time used since start of simulation (CPU-hours): " 
                    << BaseAx::getCPUTime()/3600.0;

        jobInfoFile << "\n\n";

        // plotfile information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Plotfile Information\n";
        jobInfoFile << PrettyLine;

        time_t now = std::time(0);

        // Convert now to tm struct for local timezone
        std::tm* localtm = std::localtime(&now);
        jobInfoFile   << "output data / time: " << std::asctime(localtm);

        char currentDir[FILENAME_MAX];
        if (getcwd(currentDir, FILENAME_MAX)) {
          jobInfoFile << "output dir:         " << currentDir << "\n";
        }

        jobInfoFile << "\n\n";


        // build information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Build Information\n";
        jobInfoFile << PrettyLine;

        jobInfoFile << "build date:    " << amrex::buildInfoGetBuildDate() << "\n";
        jobInfoFile << "build machine: " << amrex::buildInfoGetBuildMachine() << "\n";
        jobInfoFile << "build dir:     " << amrex::buildInfoGetBuildDir() << "\n";
        jobInfoFile << "AMReX dir:     " << amrex::buildInfoGetAMReXDir() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "COMP:          " << amrex::buildInfoGetComp() << "\n";
        jobInfoFile << "COMP version:  " << amrex::buildInfoGetCompVersion() << "\n";
                                            
        jobInfoFile << "\n";

        jobInfoFile << "C++ compiler:  " << amrex::buildInfoGetCXXName() << "\n";
        jobInfoFile << "C++ flags:     " << amrex::buildInfoGetCXXFlags() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "Fortran comp:  " << amrex::buildInfoGetFName() << "\n";
        jobInfoFile << "Fortran flags: " << amrex::buildInfoGetFFlags() << "\n";

        jobInfoFile << "\n";

        jobInfoFile << "Link flags:    " << amrex::buildInfoGetLinkFlags() << "\n";
        jobInfoFile << "Libraries:     " << amrex::buildInfoGetLibraries() << "\n";

        jobInfoFile << "\n";

        // Only use the Nyx hash if we're using comoving coords.
        // const char* githash1 = amrex::buildInfoGetGitHash(1);
        const char* githash2 = amrex::buildInfoGetGitHash(2);
        /* if (std::strlen(githash1) > 0) {
          jobInfoFile << "Nyx    git hash: " << githash1 << "\n";
        } */
        if (std::strlen(githash2) > 0) {
          jobInfoFile << "AMReX git hash:  " << githash2 << "\n";
        }

        jobInfoFile << "\n\n";

        // grid information
        jobInfoFile << PrettyLine;
        jobInfoFile << " Grid Information\n";
        jobInfoFile << PrettyLine;

        int f_lev = parent->finestLevel();

        for (int i = 0; i <= f_lev; i++)
          {
            jobInfoFile << " level: " << i << "\n";
            jobInfoFile << "   number of boxes = " << parent->numGrids(i) << "\n";
            jobInfoFile << "   maximum zones   = ";
            for (int n = 0; n < BL_SPACEDIM; n++)
              {
                jobInfoFile << parent->Geom(i).Domain().length(n) << " ";
                //jobInfoFile << parent->Geom(i).ProbHi(n) << " ";
              }
            jobInfoFile << "\n\n";
          }

        jobInfoFile << " Boundary conditions\n";
        amrex::Vector<int> lo_bc_out(BL_SPACEDIM), hi_bc_out(BL_SPACEDIM);
        amrex::ParmParse pp("ax");
        pp.getarr("lo_bc",lo_bc_out,0,BL_SPACEDIM);
        pp.getarr("hi_bc",hi_bc_out,0,BL_SPACEDIM);


        // these names correspond to the integer flags setup in the
        // Nyx_setup.cpp
        const char* names_bc[] =
          { "interior", "inflow", "outflow",
            "symmetry", "slipwall", "noslipwall" };


        jobInfoFile << "   -x: " << names_bc[lo_bc_out[0]] << "\n";
        jobInfoFile << "   +x: " << names_bc[hi_bc_out[0]] << "\n";
        if (BL_SPACEDIM >= 2) {
          jobInfoFile << "   -y: " << names_bc[lo_bc_out[1]] << "\n";
          jobInfoFile << "   +y: " << names_bc[hi_bc_out[1]] << "\n";
        }
        if (BL_SPACEDIM == 3) {
          jobInfoFile << "   -z: " << names_bc[lo_bc_out[2]] << "\n";
          jobInfoFile << "   +z: " << names_bc[hi_bc_out[2]] << "\n";
        }

        jobInfoFile << "\n\n";


        // runtime parameters
        jobInfoFile << PrettyLine;
        jobInfoFile << " Inputs File Parameters\n";
        jobInfoFile << PrettyLine;

        amrex::ParmParse::dumpTable(jobInfoFile, true);

        jobInfoFile.close();
}

void BaseAx::write_parameter_file (const std::string& dir)
{
    if (level == 0)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            std::string FileName = dir + "/the_parameters";
            std::ofstream File;
            File.open(FileName.c_str(), std::ios::out|std::ios::trunc);
            if ( ! File.good()) {
                amrex::FileOpenFailed(FileName);
            }
            File.precision(15);
            amrex::ParmParse::dumpTable(File,true);
            File.close();
        }
    }
}

void BaseAx::print_grids(const std::string & file)
{
    // File format:
    // finest level
    // n_grid at level 0
    // box 1
    // box 2
    // ...
    // n_grid at level 1
    // box 1
    // box 2
    // ...
    
    std::ofstream os(file.c_str());
    
    os << parent->finestLevel() << "\n";
    
    for (int lev = 1; lev <= parent->finestLevel() ; lev++)
    {
        const amrex::BoxArray &ba = parent->getLevel(lev).boxArray();
        os << ba.size() << "\n";

        for (int i = 0; i < ba.size(); i++)
        {
            amrex::Box bx(ba[i]);
            bx.coarsen(parent->refRatio(lev-1));
            os << bx << "\n";
        }
    }

    os.close();
}
