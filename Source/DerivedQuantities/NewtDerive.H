#ifndef _NEWT_DERIVE_
#define _NEWT_DERIVE_

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

namespace Derived
{
    // Utility functions for interpolators
    inline amrex::Box grow_box_by_one(const amrex::Box& b)
    {
        return amrex::grow(b, 1);
    }

    inline amrex::Box the_same_box(const amrex::Box& b)
    {
        return b;
    }

#ifdef __cplusplus
extern "C"
{
#endif

    // Derived gravitational potential
    void derPhiGrav
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real time, const int* bcrec, int level);

    // Derived gradient of gravitational potential
    void derGradPhi
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real time, const int* bcrec, int level);

#ifdef __cplusplus
}
#endif

}

#endif
