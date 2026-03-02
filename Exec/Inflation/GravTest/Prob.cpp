#include <AxTestGrav.H>
#include <cmath> // For mathematical operations
#include <constants_cosmo.H>

#ifdef COMOV_FULL
#include <Comoving_Full.H>
#endif

using namespace amrex;

// Initialize data in position space
// void AxTestGrav::prob_initdata_pos(const int i, const int j, const int k,
//                                    amrex::Array4<amrex::Real> const& fields,
//                                    amrex::GeometryData const& geomdata,
//                                    const amrex::GpuArray<amrex::Real,
//                                    BaseAx::max_prob_param>& prob_param) {
//     const Real* prob_lo = geomdata.ProbLo();
//     const Real* dx = geomdata.CellSize();

//     // Compute the cell's position
//     Real x = prob_lo[0] + (i + 0.5) * dx[0];
//     Real y = prob_lo[1] + (j + 0.5) * dx[1];
//     Real z = prob_lo[2] + (k + 0.5) * dx[2];

//     // Define the center and radius of the sphere
//     Real center[3] = {0.5, 0.5, 0.5}; // Center of the domain (normalized
//     coordinates) Real radius = 0.1;               // Radius of the sphere

//     // Calculate the distance from the center
//     Real dist_sq = (x - center[0]) * (x - center[0]) +
//                    (y - center[1]) * (y - center[1]) +
//                    (z - center[2]) * (z - center[2]);

//     // Initialize density based on spherical region
//     Real density_value = (dist_sq <= radius * radius) ? 1.0 : 0.0;

//     // Set the density field
//     fields(i, j, k, AxTestGrav::getField(Fields::Density)) = density_value;
// }

// // Initialize data in momentum space
// void AxTestGrav::prob_initdata_mom(const int i, const int j, const int k,
//                                    amrex::Array4<amrex::Real> const& fields,
//                                    amrex::GeometryData const& geomdata,
//                                    const amrex::GpuArray<amrex::Real,
//                                    BaseAx::max_prob_param>& prob_param) {
//     const Real* prob_lo = geomdata.ProbLo();
//     const Real* dx = geomdata.CellSize();

//     Real x = prob_lo[0] + (i + 0.5) * dx[0];
//     Real y = prob_lo[1] + (j + 0.5) * dx[1];
//     Real z = prob_lo[2] + (k + 0.5) * dx[2];

//     // Initialize gradient to zero (or any other desired initialization)
//     fields(i, j, k, getField(Fields::GradPhi)) = 0.0; // x-component
//     fields(i, j, k + 1, getField(Fields::GradPhi)) = 0.0; // y-component
//     fields(i, j, k + 2, getField(Fields::GradPhi)) = 0.0; // z-component
// }
