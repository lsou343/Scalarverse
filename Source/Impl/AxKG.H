#ifndef _AX_KG_H_
#define _AX_KG_H_

#include <BaseAx.H>

/////////////////////////////////////////////////////////////////////
//    Implementation of BaseAx to solve the KG equation for one    //
//    field in a number of potentials. No gravity,                 //
//    comoving, or dark matter physics here.                       //
//                                                                 //
//    For the same thing in a consistently-evolved comoving        //
//    background, see AxKGComov.                                   //
/////////////////////////////////////////////////////////////////////

class AxKG
    :
    public BaseAx
{
    public:
        
        ////////////////////////////////////////////////////////////
        /////////    Variables       ///////////////////////////////
        ////////////////////////////////////////////////////////////

        static int simType;
        static amrex::Real KG0;       // Need the initial value of the field.  --PH
        static amrex::Real KG_k;      // Initial fixed momentum, if using. Really, these two should be a vector, like simPars.  --PH
        static amrex::Real cutoff_k;  // Cutoff for momentum-space ICs
        static amrex::Real A;
        static amrex::Real B;
        static amrex::Real r;
        static amrex::Real beta;
        static amrex::Real s;
        static std::vector<amrex::Real> simPars; // The parameters for a given simulation type. 

        ////////////////////////////////////////////////////////////
        /////////    Functions       ///////////////////////////////
        ////////////////////////////////////////////////////////////
        
        ///////////////
        // Mandatory //
        ///////////////

        // In Impl
        AxKG();
        AxKG(amrex::Amr& papa, int lev, const amrex::Geometry& level_geom,
            const amrex::BoxArray& bl, const amrex::DistributionMapping& dm,
            amrex::Real time);
        virtual ~AxKG() {};
        virtual void initData();
        static void variable_setup();

        // In Timestep
        virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle);
        
        // In IO
        virtual void write_info ();
        virtual void writePlotFilePost(const std::string& dir, std::ostream& os);
        // virtual void writeJobInfo (const std::string& dir);

        ////////////////////////////////////////////////////////////
        /////////    Types           ///////////////////////////////
        ////////////////////////////////////////////////////////////

        ////////////////////
        // Optional/Local //
        ////////////////////
        
        //
        // Error estimation for regridding.
        //
        virtual void errorEst(amrex::TagBoxArray& tb, int clearval, int tagval, amrex::Real time,
            int n_error_buf=0, int ngrow=0);
        
        // Finally for public, keep track of the different types of particles/fields involved, and the simulation type. Must be specified by the derived class (it will only matter to them anyway). 
        enum class StateType : unsigned char  
        {
            KG_Type = 0,
        };

        enum class Fields : unsigned char 
        {
            KGf = 0,
            KGfv,
        };

        enum Sims : unsigned char
        {
            SFQ = 0,  // Single-field with quartic potential.
            GMON,     // Single-field with generalized monodromy potential à la Amin, Easther, et al.
            LOR,      // Richard's designer modified Lorentzian potential 
            TMI,      // T-model inflation
            EMI,      // E-model inflation
        };

        enum class ICType : unsigned char  
        {
            uniform = 0,
            fixed_k,
            delta_k,
            standard
        };
        
        static int nFields();
        static int getField(Fields f);
        static Fields getField(int f);
        static int nStates();
        static int getState(StateType st);
        static StateType getState(int st);
        static int getIC(ICType it);
        static ICType getIC(int it);

    protected:

        ////////////////////////////////////////////////////////////
        /////////    Variables       ///////////////////////////////
        ////////////////////////////////////////////////////////////

        int BaseAx::State_for_Time = 0;       // Must ensure the state used to define the time variable is mapped to 0 in getState().

		static ICType ic;                // Which IC to use 


        ////////////////////////////////////////////////////////////
        /////////    Functions       ///////////////////////////////
        ////////////////////////////////////////////////////////////

        ///////////////
        // Mandatory //
        ///////////////


        // In Impl
        virtual void init(amrex::AmrLevel& old);
        virtual void init();

        // This depends on the physics, so must be implemented in the derived class.  --PH
        amrex::Real est_time_step(amrex::Real dt_old);
        virtual void average_down();

        // In IO 
        static void read_params();
        static void prob_param_fill(amrex::GpuArray<amrex::Real, BaseAx::max_prob_param> &prob_params);
        
        // In Prob.cpp
        virtual void prob_initdata_pos(const int i,
                           const int j,
                           const int k,
                           amrex::Array4<amrex::Real> const& fields,
                           amrex::GeometryData const& geomdata,
                           const amrex::GpuArray<amrex::Real,BaseAx::max_prob_param>& prob_param);
   
        // initial condition at i,j,k in momentum-space. "state" contains field data (to be initialized here),
        // "geomdata" contains the problem geometry, and "prob_param" contains the relevant problem parameters.
        virtual void prob_initdata_mom(const int i,
                           const int j,
                           const int k,
                           // amrex::Array4<amrex::Real> const& fields,
                           amrex::Array4<amrex::GpuComplex<amrex::Real>> const& fields,
                           amrex::GeometryData const& geomdata,
                           const amrex::GpuArray<amrex::Real,BaseAx::max_prob_param>& prob_param);

        // void prob_initdata_mom_on_mf(amrex::MultiFab &mf ,
        //                                 amrex::Geometry const& geom,
        //                                 const amrex::GpuArray<amrex::Real,BaseAx::max_prob_param>& prob_param);

        ////////////////////
        // Optional/Local //
        ////////////////////

        // In Timestep
		void kick_KG (amrex::Real time, amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new, const amrex::Real invdeltasq);
		void drift_KG (amrex::Real dt, amrex::MultiFab&  mf_old, amrex::MultiFab&  mf_new);
        
        // In IO
        void outputPowerSpectrum(const std::string& dir);
};
#endif //AX_KG_H
