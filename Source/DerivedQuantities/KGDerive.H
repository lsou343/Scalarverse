#ifndef _KG_DERIVE_
#define _KG_DERIVE_

#include <AMReX_BLFort.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

namespace Derived
{

    // Used for the interpolators in derive_lst.addComponent
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

    // Derive the original KG field (as opposed to program units). datfab will only contain KGfpr
    void derKGf
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    // Derive the original KG field time-derivative. datfab will contain KGfpr and KGfVpr
    void derKGfv
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    // Derive the square of the original KG field. datfab will only contain KGfpr
    void derKGfdens
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    void derEdens
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    void derEgrad
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    void derEpot
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

    void derEkin
        (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int /*ncomp*/,
         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
         amrex::Real /*time*/, const int* /*bcrec*/, int level);

#ifdef __cplusplus
}
#endif

}

#endif
