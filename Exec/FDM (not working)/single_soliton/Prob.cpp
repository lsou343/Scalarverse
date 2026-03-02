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
  Real hubl = Nyx::comoving_h;
  Real meandens = 2.775e+11*pow(hubl,2)*Nyx::comoving_OmM;

  const Real* plo = geomdata.ProbLo();
  const Real* phi = geomdata.ProbHi();
  const Real* dx = geomdata.CellSize();

  Real posx = (phi[0]+plo[0])/2.0;
  Real posy = (phi[1]+plo[1])/2.0;
  Real posz = (phi[2]+plo[2])/2.0;

  Real r = std::sqrt( pow(plo[0]+(i+0.5)*dx[0]-posx,2) +
                      pow(plo[1]+(j+0.5)*dx[1]-posy,2) +
                      pow(plo[2]+(k+0.5)*dx[2]-posz,2) );
  Real rc = 1.3 * 0.012513007848917703 / std::sqrt( Nyx::m_tt*hubl*std::sqrt(Nyx::comoving_OmM) );

  axion(i,j,k,Nyx::AxRe) = std::sqrt(meandens/pow(1.0+0.091*pow(r/rc,2),8));
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
