#include <SCHDerive.H>

#include <AxSCH.H>

#ifdef __cplusplus
extern "C" {
#endif

// Derive the original SCH field (as opposed to program units). datfab will only
// contain SCHf_re_pr
void Derived::derSCHf_Re(const amrex::Box &bx, amrex::FArrayBox &derfab,
                         int dcomp, int ncomp, const amrex::FArrayBox &datfab,
                         const amrex::Geometry &geomdata, amrex::Real time,
                         const int *bcrec, int level) {

  // Solving for phi = A^{-1}a^{-r}phi_pr

  auto const dat = datfab.array();
  auto const der = derfab.array();

  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
    der(i, j, k, 0) = dat(i, j, k, AxSCH::getField(AxSCH::Fields::SCHf_Re));
  });
}

#ifdef __cplusplus
}
#endif
