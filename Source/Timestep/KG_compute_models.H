#ifndef _KG_MODELS_H_
#define _KG_MODELS_H_

// #include <AMReX_MultiFab.H>
#include <AxKG.H>
#include <AMReX_REAL.H>
#include <AMReX_Array4.H>

namespace Models
{

enum class Quant
{
    V,
    Vp,
    Vpp
};

// Returns the RHS of equations of motion, used in AmrLevel::advance(). This function computes the gradient.
amrex::Real compute_acceleration (amrex::Array4<amrex::Real> const& arr, int i, int j, int k, int comp, amrex::Real invdeltasq, amrex::Real a, amrex::Real ap, amrex::Real app);

// Returns: quantity==V   --> V
//          quantity==Vp  --> dV/df_i
//          quantity==Vpp --> d^2V/df_i^2
amrex::Real compute_model_quantity(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// V = 0.5 m^2 f^2 + 0.25 lambda f^4
amrex::Real singleFieldQuartic(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// V = (m^2 M^2 / 2 alpha)[(1 + f^2 / M^2)^alpha - 1]
amrex::Real generalisedMonodromy(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// V = 0.5 m^2 f^2 / [1 + 0.5 m^2 f^2 / M^4]
amrex::Real pseudoLorentzian(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// V = 3 alpha mass^2 tanh^2n (phi/sqrt(6 alpha))
amrex::Real Tmodel(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// V = alpha mass^2 (1 - e^(-sqrt(2/(3 alpha)) phi))^(2 n_pow)
amrex::Real Emodel(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity);

// Simply computes the un-normalized Laplacian
inline amrex::Real compute_grad2F(amrex::Array4<const amrex::Real> const& arr, int i, int j, int k, int comp, int neighbours)
{
    switch(neighbours)
    {
        case 1:
            return (
                /////// This is the 3D 9-point stencil, which is 27 points (9-points in each independent plane).  --PH
                  6.0*arr(i+1,j  ,k  ,comp)
                + 6.0*arr(i-1,j  ,k  ,comp)
                + 6.0*arr(i  ,j+1,k  ,comp)
                + 6.0*arr(i  ,j-1,k  ,comp)
                + 6.0*arr(i  ,j  ,k+1,comp)
                + 6.0*arr(i  ,j  ,k-1,comp)
                + 3.0*arr(i+1,j+1,k  ,comp)
                + 3.0*arr(i+1,j-1,k  ,comp)
                + 3.0*arr(i-1,j+1,k  ,comp)
                + 3.0*arr(i-1,j-1,k  ,comp)
                + 3.0*arr(i+1,j  ,k+1,comp)
                + 3.0*arr(i+1,j  ,k-1,comp)
                + 3.0*arr(i-1,j  ,k+1,comp)
                + 3.0*arr(i-1,j  ,k-1,comp)
                + 3.0*arr(i  ,j+1,k+1,comp)
                + 3.0*arr(i  ,j+1,k-1,comp)
                + 3.0*arr(i  ,j-1,k+1,comp)
                + 3.0*arr(i  ,j-1,k-1,comp)
                + 2.0*arr(i+1,j+1,k+1,comp)
                + 2.0*arr(i+1,j+1,k-1,comp)
                + 2.0*arr(i+1,j-1,k+1,comp)
                + 2.0*arr(i+1,j-1,k-1,comp)
                + 2.0*arr(i-1,j+1,k+1,comp)
                + 2.0*arr(i-1,j+1,k-1,comp)
                + 2.0*arr(i-1,j-1,k+1,comp)
                + 2.0*arr(i-1,j-1,k-1,comp)
                -88.0*arr(i  ,j  ,k  ,comp)
            )/26.0;
        case 2:
            return (
                /////// This is the more standard 13-point stencil, just 2 points on either side of the centre in each independent direction.  --PH
                -     arr(i-2,j  ,k  ,comp)
                +16.0*arr(i-1,j  ,k  ,comp)
                +16.0*arr(i+1,j  ,k  ,comp)
                -     arr(i+2,j  ,k  ,comp)
                -     arr(i  ,j-2,k  ,comp)
                +16.0*arr(i  ,j-1,k  ,comp)
                +16.0*arr(i  ,j+1,k  ,comp)
                -     arr(i  ,j+2,k  ,comp)
                -     arr(i  ,j  ,k-2,comp)
                +16.0*arr(i  ,j  ,k-1,comp)
                +16.0*arr(i  ,j  ,k+1,comp)
                -     arr(i  ,j  ,k+2,comp)
                -90.0*arr(i  ,j  ,k  ,comp)
            )/12.0;

        default:
            amrex::Error("Need one or two neighbours for Laplace operator");
            return -99;
    }
};

inline amrex::Real * compute_rho(amrex::Array4<const amrex::Real> const& arr, int i, int j, int k, int comp, amrex::Real invdeltasq, amrex::Real a)
{
    static amrex::Real ret[2];

    // Following LatticeEasy, we'll use 0.5 <|Grad(f)|^2> = 0.5 <-f Grad^2(f)> (boundary terms vanish because of periodic BCs)
    amrex::Real grad2F = compute_grad2F(arr, i, j, k, comp, 1)*invdeltasq;
    ret[0] = -0.5*arr(i,j,k,comp)*grad2F;

    ret[1] = compute_model_quantity({arr(i,j,k,comp)}, comp, a, 0., 0., Quant::V);

    return ret;
};

}

#endif //_KG_MODELS_H_
