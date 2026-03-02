#include "Nyx.H"
#include "Prob.H"

using namespace amrex;

void prob_param_special_fill(GpuArray<Real,max_prob_param>& prob_param)
{}

void prob_errtags_default(Vector<AMRErrorTag>& errtags)
{
  AMRErrorTagInfo info;
  errtags.push_back(AMRErrorTag(1,AMRErrorTag::GREATER,"AxDens",info));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata_state(const int i,
                         const int j,
                         const int k,
                         Array4<Real> const& axion,
                         GeometryData const& geomdata,
                         const GpuArray<Real,max_prob_param>& prob_param)
{

  Real del = 0.01;
  Real meandens = 2.775e+11*pow(Nyx::comoving_h,2)*Nyx::comoving_OmM;

  const IntVect& domlo = geomdata.Domain().smallEnd();
  const IntVect& domhi = geomdata.Domain().bigEnd();

  Real xtilde = 2.0*M_PI*(i+0.5)/ (domhi[0]-domlo[0]+1);
  Real ytilde = 2.0*M_PI*(j+0.5)/ (domhi[1]-domlo[1]+1);
  Real ztilde = 2.0*M_PI*(k+0.5)/ (domhi[2]-domlo[2]+1);

  axion(i,j,k,Nyx::AxRe) = std::sqrt((1.0-del*cos(xtilde))*meandens);
  axion(i,j,k,Nyx::AxIm) = 0.0;
  axion(i,j,k,Nyx::AxPhas) = std::atan2(axion(i,j,k,Nyx::AxIm),axion(i,j,k,Nyx::AxRe));
  axion(i,j,k,Nyx::AxDens) = axion(i,j,k,Nyx::AxRe)*axion(i,j,k,Nyx::AxRe)+axion(i,j,k,Nyx::AxIm)*axion(i,j,k,Nyx::AxIm);

}

void prob_initdata_state_on_box(const Box& bx,
                                Array4<amrex::Real> const& state,
                                GeometryData const& geomdata,
                                const GpuArray<Real,max_prob_param>& prob_param)
{
  amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                     {
                       prob_initdata_state(i, j ,k, state, geomdata, prob_param);
                     });
}
