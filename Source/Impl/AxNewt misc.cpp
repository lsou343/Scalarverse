void init_rho(amrex::MultiFab &density, amrex::MultiFab &field, amrex::Geometry const &geom)
{
  const amrex::Real *dx = geom.CellSize(); 
  const amrex::Real invdeltsq = 1.0 / dx[0] / dx[0];
  
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  // amrex::MultiFab dens_fill(ba, dm, 1 /* number of components */, 0); // LSR -- unnecessary, fillK from BaseAx is for Fourier space components (I think)
  
  for (amrex::MFIter mfi(density, false); mfi.isValid(); ++mfi) {
    amrex::Array4<amrex::Real> const &arr = field.array(mfi);
    const amrex::Box &bx = mfi.tilebox();
    const auto fab_new = density.array(mfi);

    amrex::ParallelFor(bx, [&] AMREX_GPU_DEVICE(int i, int j, int k) {
      amrex::Real tmp_grad = 0., tmp_pot = 0., tmp_kin = 0.;
#ifdef INFLATION    
      amrex::Real a = Comoving::get_comoving_a(),
                  ap = Comoving::get_comoving_ap();
#else
      amrex::Real a = 1.,	// TODO: need a better method here but will be fine for now
                  ap = 0.;
#endif
      amrex::Real H = ap / a;  // LSR -- LatticeEasy doesn't call this H since H is a'/a, not ap/a but semantic issue

      tmp_grad += (1/8.)*(
                      (arr(i+1, j, k, 0) - arr(i-1, j, k, 0))*(arr(i+1, j, k, 0) - arr(i-1, j, k, 0)) +
                      (arr(i, j+1, k, 0) - arr(i, j-1, k, 0))*(arr(i, j+1, k, 0) - arr(i, j-1, k, 0)) +
                      (arr(i, j, k+1, 0) - arr(i, j, k-1, 0))*(arr(i, j, k+1, 0) - arr(i, j, k-1, 0))
                    )*invdeltsq; // 6 point stencil in 3D - breaking on one boundary (i-direction). Suspect it's to do with the multifab box, maybe not looping around properly?
                    // Maybe we can't use mesh refinement with this? Not easily at least

      tmp_pot = Models::compute_model_quantity({arr(i,j,k,0)}, 0, a, ap, 0 /* app doesn't matter and neither does ap? So why are they included? */ , Models::Quant::V); // NEW TODO: Figure this out

      tmp_kin += 0.5*arr(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*arr(i,j,k,AxKG::getField(AxKG::Fields::KGfv));
      tmp_kin -= AxKG::r*arr(i,j,k,AxKG::getField(AxKG::Fields::KGfv))*arr(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H;
      tmp_kin += 0.5*AxKG::r*AxKG::r*arr(i,j,k,AxKG::getField(AxKG::Fields::KGf))*arr(i,j,k,AxKG::getField(AxKG::Fields::KGf))*H*H;

      const amrex::Real coef = (AxKG::B*AxKG::B/AxKG::A/AxKG::A);
      amrex::Real rho = (tmp_kin + pow(a, -2.*AxKG::s-2.)*tmp_grad + tmp_pot);
//      rho = tmp_grad;
      rho *= coef;

      fab_new(i,j,k, AxNewt::getField(AxNewt::Fields::Density)) = rho; // LSR -- this works! Now just figure out above

    });
    // Note: I'm going to keep everything in here for now but eventually I will create init_density() and init_phi() functions in Newtonian.H or Newtonian.cpp
  }
}
