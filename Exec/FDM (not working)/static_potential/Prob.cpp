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
  Real omega = 1.0;

  const Real* plo = geomdata.ProbLo();
  const Real* dx = geomdata.CellSize();

  Real x = plo[0]+(i+0.5)*dx[0];
  Real y = plo[1]+(j+0.5)*dx[1];
  Real z = plo[2]+(k+0.5)*dx[2];

  axion(i,j,k,Nyx::AxRe) = std::sqrt(std::exp(-(pow(x,2))*omega/Nyx::hbaroverm)*std::sqrt(omega/M_PI/Nyx::hbaroverm));
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
