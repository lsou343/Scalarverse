#include <KGDerive.H>

#include <AxKG.H>

#include <Comoving_Full.H>

#include <KG_compute_models.H>

#ifdef __cplusplus
extern "C"
{
#endif

void Derived::derKGf (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Solving for phi = A^{-1}a^{-r}phi_pr
    
    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        der(i,j,k,0) = dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))/(AxKG::A*std::pow(Comoving::get_comoving_a(), AxKG::r));
    });

}

void Derived::derKGfv (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{
    
    // Solving for \dot{phi} = (B/A)a^{s-r}[phi_pr^{\prime} - r a^{-1}a^{\prime} \phi_pr]

    auto const dat = datfab.array();
    auto const der = derfab.array();

    static amrex::Real BoverA  = AxKG::B/AxKG::A;
    static amrex::Real SminusR = AxKG::s-AxKG::r;

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        der(i,j,k,0) = BoverA*std::pow(Comoving::get_comoving_a(), SminusR)*(dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv)) - AxKG::r*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*Comoving::get_comoving_ap()/Comoving::get_comoving_a()) ;
    });
}

void Derived::derKGfdens (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Just return phi^2 
    
    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        der(i,j,k,0) = dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))/(AxKG::A*std::pow(Comoving::get_comoving_a(), AxKG::r));
        der(i,j,k,0) *= der(i,j,k,0);
    });

}

void Derived::derEdens (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Return rho = (B^2/A^2)a^{2s - 2r}rho_pr

    auto const dat = datfab.array();
    auto const der = derfab.array();

    const amrex::Real invdeltsq  = 1.0 / geomdata.CellSize(0) / geomdata.CellSize(0);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real tmp_grad = 0., tmp_pot = 0., tmp_kin = 0.;

        amrex::Real * tmp = Models::compute_rho(dat, i, j, k, AxKG::getField(AxKG::Fields::KGf), invdeltsq, Comoving::get_comoving_a());
        tmp_pot  = tmp[1]; 
        tmp_grad = (1/8.)*(
                      (dat(i+1,j,k,0) - dat(i-1, j, k, 0))*(dat(i+1,j,k,0) - dat(i-1, j, k, 0)) +
                      (dat(i,j+1,k,0) - dat(i, j-1, k, 0))*(dat(i,j+1,k,0) - dat(i, j-1, k, 0)) +
                      (dat(i,j,k+1,0) - dat(i, j, k-1, 0))*(dat(i,j,k+1,0) - dat(i, j, k-1, 0))
                    )*invdeltsq; 
        
        amrex::Real a = Comoving::get_comoving_a(), ap = Comoving::get_comoving_ap();
        amrex::Real H = ap/a;
        tmp_kin += 0.5*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv));
        tmp_kin -= AxKG::r*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H;
        tmp_kin += 0.5*AxKG::r*AxKG::r*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H*H;

        const amrex::Real coef = (AxKG::B*AxKG::B/AxKG::A/AxKG::A)*std::pow(a, 2.*AxKG::s-2.*AxKG::r);
        amrex::Real rho = (tmp_kin + pow(a, -2.*AxKG::s-2.)*tmp_grad + tmp_pot);
        rho *= coef;

        der(i,j,k,0) = rho;
    });
}

void Derived::derEgrad (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Return rho_g = (A^2/B^2)a^{-2s + 2r}rho_g

    auto const dat = datfab.array();
    auto const der = derfab.array();

    const amrex::Real invdeltsq  = 1.0 / geomdata.CellSize(0) / geomdata.CellSize(0);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real tmp_grad = 0.;

        tmp_grad = (1/8.)*(
                      (dat(i+1,j,k,0) - dat(i-1, j, k, 0))*(dat(i+1,j,k,0) - dat(i-1, j, k, 0)) +
                      (dat(i,j+1,k,0) - dat(i, j-1, k, 0))*(dat(i,j+1,k,0) - dat(i, j-1, k, 0)) +
                      (dat(i,j,k+1,0) - dat(i, j, k-1, 0))*(dat(i,j,k+1,0) - dat(i, j, k-1, 0))
                    )*invdeltsq; 
        
        amrex::Real a = Comoving::get_comoving_a(), ap = Comoving::get_comoving_ap();

        const amrex::Real coef = (AxKG::B*AxKG::B/AxKG::A/AxKG::A)*std::pow(a, 2.*AxKG::s-2.*AxKG::r);
        tmp_grad *= coef*pow(a, -2.*AxKG::s-2.);

        der(i,j,k,0) = tmp_grad;
    });
}

void Derived::derEpot (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Return rho_v = (A^2/B^2)a^{-2s + 2r}rho_v

    auto const dat = datfab.array();
    auto const der = derfab.array();

    const amrex::Real invdeltsq  = 1.0 / geomdata.CellSize(0) / geomdata.CellSize(0);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real tmp_pot = 0.;

        amrex::Real * tmp = Models::compute_rho(dat, i, j, k, AxKG::getField(AxKG::Fields::KGf), invdeltsq, Comoving::get_comoving_a());
        tmp_pot  = tmp[1]; 
        
        amrex::Real a = Comoving::get_comoving_a(), ap = Comoving::get_comoving_ap();

        const amrex::Real coef = (AxKG::B*AxKG::B/AxKG::A/AxKG::A)*std::pow(a, 2.*AxKG::s-2.*AxKG::r);
        tmp_pot *= coef;

        der(i,j,k,0) = tmp_pot;
    });
}

void Derived::derEkin (const amrex::Box& bx, amrex::FArrayBox& derfab, int dcomp, int ncomp,
 const amrex::FArrayBox& datfab, const amrex::Geometry& geomdata,
 amrex::Real time, const int* bcrec, int level)
{

    // Return rho_k = (A^2/B^2)a^{-2s + 2r}rho_k

    auto const dat = datfab.array();
    auto const der = derfab.array();

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        amrex::Real tmp_kin = 0.;

        amrex::Real a = Comoving::get_comoving_a(), ap = Comoving::get_comoving_ap();
        amrex::Real H = ap/a;
        tmp_kin += 0.5*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv));
        tmp_kin -= AxKG::r*dat(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H;
        tmp_kin += 0.5*AxKG::r*AxKG::r*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*dat(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H*H;

        const amrex::Real coef = (AxKG::B*AxKG::B/AxKG::A/AxKG::A)*std::pow(a, 2.*AxKG::s-2.*AxKG::r);
        tmp_kin *= coef;

        der(i,j,k,0) = tmp_kin;
    });
}

#ifdef __cplusplus
}
#endif
