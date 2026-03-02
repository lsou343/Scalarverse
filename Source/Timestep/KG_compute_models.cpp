#include <KG_compute_models.H>
#include <AxKG.H>

namespace Models
{

amrex::Real compute_acceleration (amrex::Array4<amrex::Real> const& arr, int i, int j, int k, int comp, amrex::Real invdeltasq, amrex::Real a, amrex::Real ap, amrex::Real app)
{

    amrex::Real grad2F = compute_grad2F(arr, i, j, k, comp, 1)*invdeltasq;

    amrex::Real ret = 0.;

    if (a == 0.)
    {
        ret = grad2F;

    }
    else
    {
        ret = 
            //// a^{-2s - 2}\nabla^2 f_pr
            pow(a,-2.*AxKG::s-2.)*grad2F 

            //// + (r(s-r+2)*(a'/a)^2 + r*a''/a)f_pr
            + (AxKG::r*(AxKG::s-AxKG::r+2.)*(ap/a)*(ap/a) + AxKG::r*app/a)*arr(i,j,k,comp);
    }

    ret -= compute_model_quantity({arr(i,j,k,comp)}, comp, a, ap, app, Quant::Vp);

    return ret;
}

amrex::Real compute_model_quantity(amrex::Vector<amrex::Real> f, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{
    switch (AxKG::simType)
    {
        case AxKG::SFQ:
            return singleFieldQuartic({f}, comp, a, ap, app, quantity);
        case AxKG::GMON:
            return generalisedMonodromy({f}, comp, a, ap, app, quantity);
        case AxKG::LOR:
            return pseudoLorentzian({f}, comp, a, ap, app, quantity);
        case AxKG::TMI:
            return Tmodel({f}, comp, a, ap, app, quantity);
        case AxKG::EMI:
            return Emodel({f}, comp, a, ap, app, quantity);
        default:
            amrex::Abort("Models::compute_model_quantity Error: simType is not a recognized simulation type.");
            return -1.;
    }
}

amrex::Real singleFieldQuartic(amrex::Vector<amrex::Real> ff, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{

    amrex::Real ret;

    if(AxKG::simPars.size() != 2)
        amrex::Abort("KG_compute_acceleration Error: Not enough parameters supplied for singleFieldQuartic simulation");

    static amrex::Real mass = AxKG::simPars[0]; 
    static amrex::Real lambda = AxKG::simPars[1]; 

    amrex::Real f = ff[0];

    if(a == 0.)
        a = 1.;

    switch(quantity)
    {
        // V_pr
        case(Quant::V):

            // V_pr = (B a^s)^{-2}(m^2/2) \phi_pr^2 + (A B a^{s+r})^2(lambda / 4) \phi_pr^4

            // if (mass == 0.)
            // else
                ret =
                    0.5*mass*mass*std::pow(a,-2.*AxKG::s)*f*f/(AxKG::B*AxKG::B) + 0.25*lambda*f*f*f*f/(AxKG::A*AxKG::A*AxKG::B*AxKG::B*std::pow(a,2.*AxKG::s+2.*AxKG::r));
            break;

        // \partial V_pr/ \partial f_pr
        case(Quant::Vp):

            // if (mass == 0.)
            // else 
                ret =
                    mass*mass*std::pow(a,-2.*AxKG::s)*f/(AxKG::B*AxKG::B) + lambda*f*f*f/(AxKG::A*AxKG::A*AxKG::B*AxKG::B*std::pow(a,2.*AxKG::s+2.*AxKG::r));

            break;
            
        // \partial^2 V_pr/ \partial^2 f_pr
        case(Quant::Vpp):

            // if (mass == 0.)
            // else 
                ret =
                    mass*mass*std::pow(a,-2.*AxKG::s)/(AxKG::B*AxKG::B) + 3.*lambda*f*f/(AxKG::A*AxKG::A*AxKG::B*AxKG::B*std::pow(a,2.*AxKG::s+2.*AxKG::r));
            break;

        default:
            amrex::Abort("Models::singleFieldQuartic Error: Unknown quantity requested! Must be one of Models::Quant::V, Models::Quant::Vp, or Models::Quant::Vpp.");
    }

    return ret;
}

amrex::Real generalisedMonodromy(amrex::Vector<amrex::Real> ff, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{

    amrex::Real ret;

    if(AxKG::simPars.size() != 4)
        amrex::Abort("KG_compute_acceleration Error: Not enough parameters supplied for generalisedMonodromy simulation");

    static amrex::Real mass = AxKG::simPars[0]; 
    static amrex::Real MASS = AxKG::simPars[1]; 
    static amrex::Real alph = AxKG::simPars[2]; 
    static amrex::Real d    = AxKG::simPars[3]; 

    amrex::Real f = ff[0];

    if(a == 0.)
        a = 1.;

    switch(quantity)
    {
        // V_pr
        case(Quant::V):
            ret =
                            // a^(-2s + 2r)(A^2/B^2)(m^2 M^2/2 alpha)[(1 + phi_pr^2/M^2)^alpha - 1]
                // a == 0. ? (AxKG::A*AxKG::A/(AxKG::B*AxKG::B))*(mass*mass*MASS*MASS/2./alph)*(pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A), alph) - 1.) 
                // : pow(a, 2.*AxKG::r - 2.*AxKG::s)*(AxKG::A*AxKG::A/(AxKG::B*AxKG::B))*(mass*mass*MASS*MASS/2./alph)*(pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph) - 1.);
                pow(a, 2.*AxKG::r - 2.*AxKG::s)*(AxKG::A*AxKG::A/(AxKG::B*AxKG::B))*(mass*mass*MASS*MASS/2./alph)*(pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph) - 1.);
            if (d != 0)
                ret += (d*mass*mass*(1. - alph))*f*f*f*f/(4.*AxKG::A*AxKG::A*AxKG::B*AxKG::B*pow(a, -2.*AxKG::s - 2.*AxKG::r));
            break;
            
        // \partial V_pr/ \partial f_pr
        case(Quant::Vp):
            ret = 
                // a == 0. ? (1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A), alph - 1.)*f   // Minkowski
                // : pow(a, - 2.*AxKG::s)*(1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 1.)*f;  // Friedmann
                pow(a, - 2.*AxKG::s)*(1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 1.)*f;  // Friedmann
            if (d != 0)
                ret += (d*mass*mass*(1. - alph))*f*f*f/(AxKG::A*AxKG::A*AxKG::B*AxKG::B*pow(a, -2.*AxKG::s - 2.*AxKG::r));
            break;

        // \partial^2 V_pr/ \partial^2 f_pr
        case(Quant::Vpp):
            ret = 
                // a == 0. ? (1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A), alph - 1.)                                     // Minkowski
                //             + (2.*(alph-1.)/(MASS*MASS*AxKG::A*AxKG::A*AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A), alph - 2.)*f*f
                // : pow(a, - 2.*AxKG::s)*(1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 1.)  // Friedmann
                //     + pow(a, - 2.*AxKG::s - 2.*AxKG::r)*(2.*(alph-1.)/(MASS*MASS*AxKG::A*AxKG::A*AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 2.)*f*f;
                pow(a, - 2.*AxKG::s)*(1./(AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 1.)  // Friedmann
                    + pow(a, - 2.*AxKG::s - 2.*AxKG::r)*(2.*(alph-1.)/(MASS*MASS*AxKG::A*AxKG::A*AxKG::B*AxKG::B))*(mass*mass)*pow(1. + f*f/(MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r)), alph - 2.)*f*f;
            if (d != 0)
                ret += 3.*(d*mass*mass*(1. - alph))*f*f/(AxKG::A*AxKG::A*AxKG::B*AxKG::B*pow(a, -2.*AxKG::s - 2.*AxKG::r));
            break;

        default:
            amrex::Abort("Models::generalisedMonodromy Error: Unknown quantity requested! Must be one of Models::Quant::V, Models::Quant::Vp, or Models::Quant::Vpp.");
    }

    return ret;
}

amrex::Real pseudoLorentzian(amrex::Vector<amrex::Real> ff, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{

    amrex::Real ret;

    if(AxKG::simPars.size() != 2)
        amrex::Abort("KG_compute_acceleration Error: Not enough parameters supplied for generalisedMonodromy simulation");

    static amrex::Real mass = AxKG::simPars[0]; 
    static amrex::Real MASS = AxKG::simPars[1]; 

    amrex::Real f = ff[0];

    if(a == 0.)
        a = 1.;

    amrex::Real C1 = mass*mass/(2.*AxKG::B*AxKG::B*pow(a, 2.*AxKG::s));
    amrex::Real C2 = mass*mass/(2.*MASS*MASS*MASS*MASS*AxKG::A*AxKG::A*pow(a, 2.*AxKG::r));

    switch(quantity)
    {
        // V_pr
        case(Quant::V):
            ret =
                // a^(-2s + 2r)(A^2/B^2)(0.5 m^2 (phi_pr / A a^r)^2)/(1 + m^2 phi_pr^2 / 2 M^4 A^2 a^2r)
                C1*(f*f)/(1. + C2*f*f);
            break;
            
        // \partial V_pr/ \partial f_pr
        case(Quant::Vp):
            ret = 
                2.*C1*f/(1 + C2*f*f) - 2.*C1*C2*f*f*f/((1. + C2*f*f)*(1. + C2*f*f));
            break;

        // \partial^2 V_pr/ \partial^2 f_pr
        case(Quant::Vpp):
            ret = 
                2.*C1/(1 + C2*f*f) - 10.*C1*C2*f*f/((1. + C2*f*f)*(1. + C2*f*f)) + 8.*C1*C2*C2*f*f*f*f/((1. + C2*f*f)*(1. + C2*f*f)*(1. + C2*f*f));
            break;

        default:
            amrex::Abort("Models::pseudoLorentzian Error: Unknown quantity requested! Must be one of Models::Quant::V, Models::Quant::Vp, or Models::Quant::Vpp.");
    }

    return ret;
}
amrex::Real Tmodel(amrex::Vector<amrex::Real> ff, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{

    amrex::Real ret;

    if(AxKG::simPars.size() != 3)
        amrex::Abort("KG_compute_acceleration Error: Not enough parameters supplied for Tmodel simulation");

    static amrex::Real mass = AxKG::simPars[0]; 
    static amrex::Real alpha = AxKG::simPars[1];
    static amrex::Real n_pow = AxKG::simPars[2];

    amrex::Real f = ff[0];

    if(a == 0.)
        a = 1.;

    switch(quantity)
    {
        // V_pr
        case(Quant::V):

            // V_pr = 3 alpha mass^2 tanh^2n (phi/sqrt(6 alpha))

            // if (mass == 0.)
            // else
                ret = pow(sqrt(6*alpha) * AxKG::A, 2*n_pow) * pow(a, -2*AxKG::s + 2*AxKG::r) * pow(std::tanh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))), 2*n_pow) / (2*n_pow);
            break;

        // \partial V_pr/ \partial f_pr
        case(Quant::Vp):

            // if (mass == 0.)
            // else 
                ret = pow(sqrt(6*alpha) * AxKG::A, 2*n_pow-1) * pow(a, -2*AxKG::s + AxKG::r) * pow(std::tanh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))), 2*n_pow-1) / pow(std::cosh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))), 2);
            break;
            
        // \partial^2 V_pr/ \partial^2 f_pr
        case(Quant::Vpp):

            // if (mass == 0.)
            // else 
            	  ret = pow(sqrt(6*alpha) * AxKG::A, 2*n_pow-2) * pow(a, -2*AxKG::s) * ((2*n_pow-1)*pow(std::tanh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))),2*n_pow-2) / pow(std::cosh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))),4) - 2 * pow(std::tanh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))),2*n_pow) / pow(std::cosh(f / (AxKG::A * sqrt(6*alpha) * pow(a, AxKG::r))),2));
            break;

        default:
            amrex::Abort("Models::Tmodel Error: Unknown quantity requested! Must be one of Models::Quant::V, Models::Quant::Vp, or Models::Quant::Vpp.");
    }

    return ret;
}
amrex::Real Emodel(amrex::Vector<amrex::Real> ff, int comp, amrex::Real a, amrex::Real ap, amrex::Real app, Quant quantity)
{

    amrex::Real ret;

    if(AxKG::simPars.size() != 3)
        amrex::Abort("KG_compute_acceleration Error: Not enough parameters supplied for E-model simulation");

    static amrex::Real mass = AxKG::simPars[0]; 
    static amrex::Real alpha = AxKG::simPars[1];
    static amrex::Real n_pow = AxKG::simPars[2];

    amrex::Real f = ff[0];

    if(a == 0.)
        a = 1.;

    switch(quantity)
    {
        // V_pr
        case(Quant::V):

            // V = alpha mass^2 (1 - e^(-sqrt(2/(3 alpha)) phi))^(2 n_pow)

            // if (mass == 0.)
            // else
                ret = pow(a, -2*AxKG::s+2*AxKG::r) / (2*n_pow) * pow(1.5 * alpha * AxKG::A * AxKG::A, n_pow) * pow(1 - std::exp(-sqrt(2./(3.*alpha)) * f / (AxKG::A * pow(a, AxKG::r))), 2*n_pow);
            break;

        // \partial V_pr/ \partial f_pr
        case(Quant::Vp):

            // if (mass == 0.)
            // else 
                ret = pow(a, -2*AxKG::s+AxKG::r) * pow(sqrt(1.5*alpha)*AxKG::A, 2*n_pow-1) * std::exp(-sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r))) * pow(1-std::exp(-sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r))), 2*n_pow-1);
            break;
            
        // \partial^2 V_pr/ \partial^2 f_pr
        case(Quant::Vpp):

            // if (mass == 0.)
            // else 
            	  ret = pow(a, -2*AxKG::s) * pow(sqrt(1.5*alpha)*AxKG::A, 2*n_pow-2) * (-std::exp(-sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r))) * pow(1 - std::exp(-sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r))), 2*n_pow-1) + (2*n_pow-1) * std::exp(-2.*sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r)))*pow(1-std::exp(-sqrt(2/(3*alpha)) * f / (AxKG::A * pow(a, AxKG::r))), 2*n_pow-2));
            break;

        default:
            amrex::Abort("Models::Emodel Error: Unknown quantity requested! Must be one of Models::Quant::V, Models::Quant::Vp, or Models::Quant::Vpp.");
    }

    return ret;
}
}
