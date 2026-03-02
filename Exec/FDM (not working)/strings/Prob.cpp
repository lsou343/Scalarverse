#include "Nyx.H"
#include "Prob.H"

using namespace amrex;

void prob_param_special_fill(GpuArray<Real,max_prob_param>& prob_param)
{}

void prob_errtags_default(Vector<AMRErrorTag>& errtags)
{
  AMRErrorTagInfo info;
  errtags.push_back(AMRErrorTag(1,AMRErrorTag::GREATER,"axion_string",info));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void prob_initdata_state(const int i,
                         const int j,
                         const int k,
                         Array4<Real> const& axion,
                         GeometryData const& geomdata,
                         const GpuArray<Real,max_prob_param>& prob_param)
{
  Real r = Random();
  Real phase = 2.0*M_PI*r;
  axion(i,j,k,Nyx::AxRe)   = std::cos(phase);
  axion(i,j,k,Nyx::AxIm)   = std::sin(phase);
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
