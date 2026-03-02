#ifndef _BASE_KG_H_
#define _BASE_KG_H_

#include <AxKG.H>

/////////////////////////////////////////////////////////////////////
//    Base class to use to include KG in a derived class (e.g., if //
//    adding comoving coords or Poisson gravity).                  //
//                                                                 //
//    Only derive from this class, not AxKG!                       //
//    Note, however, that you will have access to everything in    //
//    AxKG.
/////////////////////////////////////////////////////////////////////

class BaseKG 
    :
    public AxKG
{
    public:
        
        ////////////////////////////////////////////////////////////
        /////////    Functions       ///////////////////////////////
        ////////////////////////////////////////////////////////////
        
        ///////////////
        // Mandatory //
        ///////////////

        // In Impl
        BaseKG() : AxKG() {};
        BaseKG(amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
            amrex::Real time) 
            : AxKG(papa, lev, level_geom, bl, dm, time) {};
        virtual ~BaseKG() {};

        virtual void initData() = 0;
        // THIS MUST BE IMPLEMENTED AS WELL
        // static void variable_setup(); 

        // In Timestep
        virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle) = 0;
        
        // In IO
        virtual void write_info () = 0;
        // virtual void writePlotFilePost(const std::string& dir, std::ostream& os);
        // virtual void writeJobInfo (const std::string& dir);

        ////////////////////////////////////////////////////////////
        /////////    Types           ///////////////////////////////
        ////////////////////////////////////////////////////////////

        ////////////////////
        // Optional/Local //
        ////////////////////
        
        // THESE MUST BE RE-IMPLEMENTED IF THERE ARE MORE FIELDS
        // NOTE THAT YOU WILL NEED MORE THAN THESE IF YOU HAVE ADDITIONAL FIELDS/ICs
        // static int nFields();
        // static int getField(AxKG::Fields f);
        // static AxKG::Fields getField(int f);
        // static int nStates();
        // static int getState(AxKG::StateType st);
        // static AxKG::StateType getState(int st);
        // static int getIC(AxKG::ICType it);
        // static AxKG::ICType getIC(int it);
        
    protected:

        ///////////////
        // Mandatory //
        ///////////////


        // In Impl
        virtual void init(amrex::AmrLevel& old) = 0;
        virtual void init() = 0;

        // This depends on the physics, so must be implemented in the derived class.  --PH
        amrex::Real est_time_step(amrex::Real dt_old) = 0;
        virtual void average_down() = 0;

        // In IO 
        // THESE NEED TO BE INCLUDED, BUT STATIC FUNCTIONS CAN'T BE FORCED IN C++, SO HEADS-UP, READ THE COMMENTS
        // static void read_params();  // One sec
        // static void prob_param_fill(amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_params); // One Sec
        
        ////////////////////
        // Optional/Local //
        ////////////////////

        // In Timestep
        // PROBABLY NEED TO OVERRIDE THESE, BUT THEY ARE TECHNICALLY OPTIONAL
		// void kick_KG (amrex::Real time, amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq);
		// void drift_KG (amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new);

};
#endif //BASE_KG_H
