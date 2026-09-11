#include <NewtDerive.H>
#include <AxNewt.H>
#include <AxKG.H>
#include <Comoving_Full.H>

#ifdef __cplusplus
extern "C"
{
#endif

// void derlogden(const Box& bx, FArrayBox& derfab, int /*dcomp*/, int /*ncomp*/,
//                 const FArrayBox& datfab, const Geometry& /*geomdata*/,
//                 Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const dat = datfab.array();
//     auto const der = derfab.array();

//     amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//     {
//         der(i,j,k,0) = std::log10(dat(i,j,k,0));
//     });
// }

// void derstate(const Box& bx, FArrayBox& derfab, int /*dcomp*/, int /*ncomp*/,
//                 const FArrayBox& datfab, const Geometry& /*geomdata*/,
//                 Real /*time*/, const int* /*bcrec*/, int /*level*/)
// {
//     auto const dat = datfab.array();
//     auto const der = derfab.array();

//     amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
//         // density
//         der(i,j,k,0) = dat(i,j,k,0);

//         // temperature
//         der(i,j,k,1) = dat(i,j,k,1);

//         if(der.nComp()>=3 && dat.nComp()>=3)
//         {
//             // (rho X)_1 = X_1
//             der(i,j,k,2) = dat(i,j,k,2) / dat(i,j,k,0);
//         }
//     });
// }

void Derived::derPhiGrav(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                         amrex::Real time, const int* bcrec, int level)
{
    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::Real coef = AxKG::A * AxKG::A;
#ifdef COMOV_FULL
    coef *= std::pow(Comoving::get_comoving_a(), -2.*AxKG::s + 2.*AxKG::r);
#endif

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        der(i, j, k, dcomp) = dat(i, j, k, AxNewt::getField(AxNewt::Fields::PhiGrav)) / coef;
    });
}

void Derived::derPhiGravv(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                         amrex::Real time, const int* bcrec, int level)  // TODO: Set correct coefficient to account for time derivative
{
    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::Real coef = AxKG::B;
    coef /= AxKG::A * AxKG::A;
#ifdef COMOV_FULL
    coef *= std::pow(Comoving::get_comoving_a(), 3.*AxKG::s - 2.*AxKG::r);
    amrex::Real H = Comoving::get_comoving_ap() / Comoving::get_comoving_a();
#else
    amrex::Real H = 0.;
#endif

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real Phi_dot = dat(i, j, k, AxNewt::getField(AxNewt::Fields::PhiGravv)) + 2. * (AxKG::s - AxKG::r) * H * dat(i, j, k, AxNewt::getField(AxNewt::Fields::PhiGrav));
        der(i, j, k, dcomp) = Phi_dot / coef;
    });
}

void Derived::derGradPhi(const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
                         const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
                         amrex::Real time, const int* bcrec, int level)
{
    auto const dat = datfab.array();
    auto const der = derfab.array();
    const amrex::Real dx = geomdata.CellSize(0);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        der(i, j, k, dcomp)     = (dat(i + 1, j, k, AxNewt::getField(AxNewt::Fields::PhiGrav)) -
                                    dat(i - 1, j, k, AxNewt::getField(AxNewt::Fields::PhiGrav))) / (2.0 * dx);
        der(i, j, k, dcomp + 1) = (dat(i, j + 1, k, AxNewt::getField(AxNewt::Fields::PhiGrav)) -
                                    dat(i, j - 1, k, AxNewt::getField(AxNewt::Fields::PhiGrav))) / (2.0 * dx);
        der(i, j, k, dcomp + 2) = (dat(i, j, k + 1, AxNewt::getField(AxNewt::Fields::PhiGrav)) -
                                    dat(i, j, k - 1, AxNewt::getField(AxNewt::Fields::PhiGrav))) / (2.0 * dx);
    });
}

#ifdef __cplusplus
}
#endif
