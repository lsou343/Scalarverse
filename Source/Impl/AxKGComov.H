#ifndef _AX_KG_COMOV_H_
#define _AX_KG_COMOV_H_

#include <BaseKG.H>

/////////////////////////////////////////////////////////////////////
//    Implementation of BaseKG to solve the KG equation for one    //
//    in a consistently-evolved comoving background.               //
/////////////////////////////////////////////////////////////////////

class AxKGComov
    :
    public BaseKG
{
    public:
        
        ////////////////////////////////////////////////////////////
        /////////    Variables       ///////////////////////////////
        ////////////////////////////////////////////////////////////

  //       static int simType;
		// static amrex::Real KG0;  //Need the initial value of the field.  --PH
		// static amrex::Real KG_k;  //Initial fixed momentum, if using. Really, these two should be a vector, like simPars.  --PH
		// static amrex::Real A;
		// static amrex::Real B;
		// static amrex::Real r;
		// static amrex::Real beta;
		// static amrex::Real s;
  //       static std::vector<amrex::Real> simPars; // The parameters for a given simulation type. 

        ////////////////////////////////////////////////////////////
        /////////    Functions       ///////////////////////////////
        ////////////////////////////////////////////////////////////
        
        ///////////////
        // Mandatory //
        ///////////////

        // In Impl
        AxKGComov() : BaseKG() {};
        AxKGComov(amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
            amrex::Real time)
            : BaseKG(papa, lev, level_geom, bl, dm, time) {};
        virtual ~AxKGComov();

        virtual void initData() override; // LSR -- added override
        // THIS MUST BE IMPLEMENTED AS WELL
        static void variable_setup(); 

        // In Timestep
        virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle);
        amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle, int stage); 
        
        // In IO
        virtual void write_info ();
        virtual void writePlotFilePost(const std::string& dir, std::ostream& os);
        virtual void checkPointPost (const std::string& dir, std::ostream& os);

        // virtual void writeJobInfo (const std::string& dir);

        ////////////////////////////////////////////////////////////
        /////////    Types           ///////////////////////////////
        ////////////////////////////////////////////////////////////

        ////////////////////
        // Optional/Local //
        ////////////////////
        void fill_rho();

        // We will maybe need to update the grid points in Comoving upon re-gridding
        // virtual void post_regrid (int lbase, int new_finest);
        
        // THESE MUST BE RE-IMPLEMENTED IF THERE ARE MORE FIELDS
        // NOTE THAT YOU WILL NEED MORE THAN THESE IF YOU HAVE ADDITIONAL FIELDS/ICs
        static int nFields() { return AxKG::nFields(); } ;
        static int getField(AxKG::Fields f) { return AxKG::getField(f); };
        static AxKG::Fields getField(int f){ return AxKG::getField(f); };
        static int nStates(){ return AxKG::nStates(); };
        static int getState(AxKG::StateType st){ return AxKG::getState(st); };
        static AxKG::StateType getState(int st){ return AxKG::getState(st); };
        static int getIC(AxKG::ICType it){ return AxKG::getIC(it); };
        static AxKG::ICType getIC(int it){ return AxKG::getIC(it); };
        
    protected:

        ///////////////
        // Mandatory //
        ///////////////


        // In Impl
        virtual void init(amrex::AmrLevel& old);
        virtual void init();
        virtual void post_restart();

        // This depends on the physics, so must be implemented in the derived class.  --PH
        amrex::Real est_time_step(amrex::Real dt_old);
        virtual void average_down();

        // In IO 
        // THESE NEED TO BE INCLUDED, BUT STATIC FUNCTIONS CAN'T BE FORCED IN C++, SO HEADS-UP, READ THE COMMENTS
        static void read_params(); 
        
        ////////////////////
        // Optional/Local //
        ////////////////////

        // In Impl
        void fill_rho(amrex::MultiFab &mf, amrex::Real invdelsq);

        // In Timestep
        void kick_KG (amrex::Real time, amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq);
        void drift_KG (amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new);
};
#endif //AX_KG_COMOV_H
