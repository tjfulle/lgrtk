#ifndef LGR_HYPER_EP_MODEL_RT_HPP
#define LGR_HYPER_EP_MODEL_RT_HPP

#include <string>
#include <limits>
#include <algorithm>
#include <iostream>  // DEBUGGING

#include <hpc_symmetric3x3.hpp>
#include "common.hpp"

namespace lgr {
namespace hyper_ep {

HPC_NOINLINE inline
double
flow_stress(Properties props, double const temp, double const ep,
    double const epdot, double const dp)
{
  auto Y = std::numeric_limits<double>::max();
  if (props.hardening == Hardening::NONE) {
    Y = props.A;
  }
  else if (props.hardening == Hardening::LINEAR_ISOTROPIC) {
    Y = props.A + props.B * ep;
  }
  else if (props.hardening == Hardening::POWER_LAW) {
    auto const a = props.A;
    auto const b = props.B;
    auto const n = props.n;
    Y = (ep > 0.0) ? (a + b * std::pow(ep, n)) : a;
  }
  else if (props.hardening == Hardening::ZERILLI_ARMSTRONG) {
    auto const a = props.A;
    auto const b = props.B;
    auto const n = props.n;
    Y = (ep > 0.0) ? (a + b * std::pow(ep, n)) : a;
    auto const C1 = props.C1;
    auto const C2 = props.C2;
    auto const C3 = props.C3;
    auto alpha = C3;
    if (props.rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      auto const C4 = props.C4;
      alpha -= C4 * std::log(epdot);
    }
    Y += (C1 + C2 * std::sqrt(ep)) * std::exp(-alpha * temp);
  }
  else if (props.hardening == Hardening::JOHNSON_COOK) {
    auto const ajo = props.A;
    auto const bjo = props.B;
    auto const njo = props.n;
    auto const temp_ref = props.C1;
    auto const temp_melt = props.C2;
    auto const mjo = props.C3;
    // Constant contribution
    Y = ajo;
    // Plastic strain contribution
    if (bjo > 0.0) {
      Y += (std::abs(njo) > 0.0) ? bjo * std::pow(ep, njo) : bjo;
    }
    // Temperature contribution
    if (std::abs(temp_melt - std::numeric_limits<double>::max()) + 1.0 !=
        1.0) {
      auto const tstar = (temp > temp_melt)
                             ? 1.0
                             : ((temp - temp_ref) / (temp_melt - temp_ref));
      Y *= (tstar < 0.0) ? (1.0 - tstar) : (1.0 - std::pow(tstar, mjo));
    }
  }
  if (props.rate_dep == RateDependence::JOHNSON_COOK) {
    auto const cjo = props.C4;
    auto const epdot0 = props.ep_dot_0;
    auto const rfac = epdot / epdot0;
    // FIXME: This assumes that all the
    // strain rate is plastic.  Should
    // use actual strain rate.
    // Rate of plastic strain contribution
    if (cjo > 0.0) {
      Y *= (rfac < 1.0) ? std::pow((1.0 + rfac), cjo)
                        : (1.0 + cjo * std::log(rfac));
    }
  }
  return (1 - dp) * Y;
}

HPC_NOINLINE inline
double
dflow_stress(Properties const props, double const temp, double const ep,
    double const epdot, double const dtime, double const dp)
{
  double deriv = 0.;
  if (props.hardening == Hardening::LINEAR_ISOTROPIC) {
    auto const b = props.B;
    deriv = b;
  }
  else if (props.hardening == Hardening::POWER_LAW) {
    auto const b = props.B;
    auto const n = props.n;
    deriv = (ep > 0.0) ? b * n * std::pow(ep, n - 1) : 0.0;
  }
  else if (props.hardening == Hardening::ZERILLI_ARMSTRONG) {
    auto const b = props.B;
    auto const n = props.n;
    deriv = (ep > 0.0) ? b * n * std::pow(ep, n - 1) : 0.0;
    auto const C1 = props.C1;
    auto const C2 = props.C2;
    auto const C3 = props.C3;
    auto alpha = C3;
    if (props.rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      auto const C4 = props.C4;
      alpha -= C4 * std::log(epdot);
    }
    deriv +=
        .5 * C2 / std::sqrt(ep <= 0.0 ? 1.e-8 : ep) * std::exp(-alpha * temp);
    if (props.rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      auto const C4 = props.C4;
      auto const term1 = C1 * C4 * temp * std::exp(-alpha * temp);
      auto const term2 = C2 * sqrt(ep) * C4 * temp * std::exp(-alpha * temp);
      deriv += (term1 + term2) / (epdot <= 0.0 ? 1.e-8 : epdot) / dtime;
    }
  }
  else if (props.hardening == Hardening::JOHNSON_COOK) {
    auto const bjo = props.B;
    auto const njo = props.n;
    auto const temp_ref = props.C1;
    auto const temp_melt = props.C2;
    auto const mjo = props.C3;
    // Calculate temperature contribution
    double temp_contrib = 1.0;
    if (std::abs(temp_melt - std::numeric_limits<double>::max()) + 1.0 !=
        1.0) {
      auto const tstar =
          (temp > temp_melt) ? 1.0 : (temp - temp_ref) / (temp_melt - temp_ref);
      temp_contrib =
          (tstar < 0.0) ? (1.0 - tstar) : (1.0 - std::pow(tstar, mjo));
    }
    deriv =
        (ep > 0.0) ? (bjo * njo * std::pow(ep, njo - 1) * temp_contrib) : 0.0;
    if (props.rate_dep == RateDependence::JOHNSON_COOK) {
      auto const ajo = props.A;
      auto const cjo = props.C4;
      auto const epdot0 = props.ep_dot_0;
      auto const rfac = epdot / epdot0;
      // Calculate strain rate contribution
      auto const term1 = (rfac < 1.0) ? (std::pow((1.0 + rfac), cjo))
                                      : (1.0 + cjo * std::log(rfac));
      auto term2 = (ajo + bjo * std::pow(ep, njo)) * temp_contrib;
      if (rfac < 1.0) {
        term2 *= cjo * std::pow((1.0 + rfac), (cjo - 1.0));
      } else {
        term2 *= cjo / rfac;
      }
      deriv *= term1;
      deriv += term2 / dtime;
    }
  }
  constexpr double sq23 = 0.8164965809277261;
  return (1. - dp) * sq23 * deriv;
}

HPC_NOINLINE inline
double
scalar_damage(Properties const props, hpc::symmetric_stress<double>& T, double const dp,
    double const temp, double const /* ep */, double const epdot,
    double const dtime)
{
  if (props.damage == Damage::NONE) {
    return 0.0;
  }
  else if (props.damage == Damage::JOHNSON_COOK) {
    double tolerance = 1e-10;
    auto const I = hpc::symmetric_stress<double>::identity();
    auto const T_mean = (trace(T) / 3.0);
    auto const S = T - I * T_mean;
    auto const norm_S = norm(S);
    auto const S_eq = std::sqrt(norm_S * norm_S * 1.5);

    double eps_f = props.eps_f_min;
    double sig_star = (std::abs(S_eq) > 1e-16) ? T_mean / S_eq : 0.0;
    if (sig_star < 1.5) {
      // NOT SPALL
      // sig_star < 1.5 indicates spall conditions are *not* met and the failure
      // strain must be calculated.
      sig_star = std::max(std::min(sig_star, 1.5), -1.5);

      // Stress contribution to damage
      double stress_contrib = props.D1 + props.D2 * exp(props.D3 * sig_star);

      // Strain rate contribution to damage
      double dep_contrib = 1.0;
      if (epdot < 1.0) {
        dep_contrib = std::pow((1.0 + epdot), props.D4);
      } else {
        dep_contrib = 1.0 + props.D4 * std::log(epdot);
      }

      double temp_contrib = 1.0;
      auto const temp_ref = props.D6;
      auto const temp_melt = props.D7;
      if (std::abs(temp_melt-std::numeric_limits<double>::max())+1.0 != 1.0) {
        auto const tstar =
            temp > temp_melt ? 1.0 : (temp - temp_ref) / (temp_melt - temp_ref);
        temp_contrib += props.D5 * tstar;
      }

      // Calculate the updated scalar damage parameter
      eps_f = stress_contrib * dep_contrib * temp_contrib;
    }

    if (eps_f < tolerance) return dp;

    // Calculate plastic strain increment
    auto const dep = epdot * dtime;
    auto const ddp = dep / eps_f;
    return (dp + ddp < tolerance) ? 0.0 : dp + ddp;
  }

  // Should never get here. The input reader already through threw if there was
  // a bad input.
  return 0.0;
}

/* Computes the radial return
 *
 * Yield function:
 *   S:S - Sqrt[2/3] * Y = 0
 * where S is the stress deviator.
 *
 * Equivalent plastic strain:
 *   ep = Integrate[Sqrt[2/3]*Sqrt[epdot:epdot], 0, t]
 *
 */
HPC_NOINLINE inline
ErrorCode
radial_return(Properties const props, hpc::symmetric_stress<double> const Te,
    hpc::deformation_gradient<double> const F, double const temp, double const dtime, hpc::symmetric_stress<double>& T,
    hpc::deformation_gradient<double>& Fp, double& ep, double& epdot, double& dp, StateFlag& flag)
{
  constexpr double tol1 = 1e-12;
  auto const tol2 = std::min(dtime, 1e-6);
  constexpr double twothird = 2.0 / 3.0;
  auto const sq2 = std::sqrt(2.0);
  auto const sq3 = std::sqrt(3.0);
  auto const sq23 = sq2 / sq3;
  auto const sq32 = 1.0 / sq23;
  auto const E = props.E;
  auto const Nu = props.Nu;
  auto const mu = E / 2.0 / (1.0 + Nu);
  auto const twomu = 2.0 * mu;
  auto gamma = epdot * dtime * sq32;
  // Possible states at this point are TRIAL or REMAPPED
  if (flag != StateFlag::REMAPPED) flag = StateFlag::TRIAL;
  // check yield
  auto Y = flow_stress(props, temp, ep, epdot, dp);
  auto const S0 = deviatoric_part(Te);
  auto const norm_S0 = norm(S0);
  auto f = norm_S0 / sq2 - Y / sq3;
  if (f <= tol1) {
    // Elastic loading
    T = 1. * Te;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::ELASTIC;
  } else {
    int conv = 0;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::PLASTIC;
    auto const N = S0 / norm_S0;  // Flow direction
    for (int iter = 0; iter < 100; ++iter) {
      // Compute the yield stress
      Y = flow_stress(props, temp, ep, epdot, dp);
      // Compute g
      auto const g = norm_S0 - sq23 * Y - twomu * gamma;
      // Compute derivatives of g
      auto const dydg = dflow_stress(props, temp, ep, epdot, dtime, dp);
      auto const dg = -twothird * dydg - twomu;
      // Update dgamma
      auto const dgamma = -g / dg;
      gamma += dgamma;
      // Update state
      auto const dep = std::max(sq23 * gamma, 0.0);
      epdot = dep / dtime;
      ep += dep;
      auto const S = S0 - twomu * gamma * N;
      f = norm(S) / sq2 - Y / sq3;
      if (f < tol1) {
        conv = 1;
        break;
      } else if (std::abs(dgamma) < tol2) {
        conv = 1;
        break;
      } else if (iter > 24 && f <= tol1 * 1000.0) {
        // Weaker convergence
        conv = 2;
        break;
      }
#ifdef LGR_HYPER_EP_VERBOSE_DEBUG
      std::cout << "Iteration: " << iter + 1 << "\n"
                << "\tROOTJ20: " << hpc::norm(S0) << "\n"
                << "\tROOTJ2: " << hpc::norm(S) << "\n"
                << "\tep: " << ep << "\n"
                << "\tepdot: " << epdot << "\n"
                << "\tgamma: " << gamma << "\n"
                << "\tg: " << g << "\n"
                << "\tdg: " << dg << "\n\n\n";
#endif
    }
    // Update the stress tensor
    T = Te - twomu * gamma * N;
    if (!conv) {
      return ErrorCode::RADIAL_RETURN_FAILURE;
    } else if (conv == 2) {
      // print warning about weaker convergence
    }
    // Update damage
    dp = scalar_damage(props, T, dp, temp, ep, epdot, dtime);
  }

  if (flag != StateFlag::ELASTIC) {
    // determine elastic deformation
    auto const jac = determinant(F);
    auto const Bbe = find_bbe(T, mu);
    auto const j13 = std::cbrt(jac);
    auto const j23 = j13 * j13;
    auto const Be = Bbe * j23;
    auto const Ve = sqrt_spd(Be);
    Fp = inverse(Ve) * F;
    if (flag == StateFlag::REMAPPED) {
      // Correct pressure term
      auto p = trace(T);
      auto const D1 = 6.0 * (1.0 - 2.0 * Nu) / E;
      p = (2.0 * jac / D1 * (jac - 1.0)) - (p / 3.0);
      for (int i = 0; i < 3; ++i) T(i, i) = p;
    }
  }
  return ErrorCode::SUCCESS;
}

HPC_NOINLINE inline
hpc::symmetric_stress<double>
linear_elastic_stress(Properties const props, hpc::deformation_gradient<double> const Fe)
{
  auto const E = props.E;
  auto const nu = props.Nu;
  auto const K = E / (3.0 * (1.0 - 2.0 * nu));
  auto const G = E / 2.0 / (1.0 + nu);
  auto const grad_u = Fe - hpc::deformation_gradient<double>::identity();
  auto const strain = symmetric_part(grad_u);
  return (3.0 * K) * isotropic_part(strain) + (2.0 * G) * deviatoric_part(strain);
}

/*
 * Update the stress using Neo-Hookean hyperelasticity
 *
 */
HPC_NOINLINE inline
hpc::symmetric_stress<double>
hyper_elastic_stress(Properties const props, hpc::deformation_gradient<double> const Fe, double const jac)
{
  auto const E = props.E;
  auto const Nu = props.Nu;
  // Jacobian and distortion tensor
  auto const scale = 1.0 / std::cbrt(jac);
  auto const Fb = scale * Fe;
  // Elastic moduli
  auto const C10 = E / (4.0 * (1.0 + Nu));
  auto const D1 = 6.0 * (1.0 - 2.0 * Nu) / E;
  auto const EG = 2.0 * C10 / jac;
  // Deviatoric left Cauchy-Green deformation tensor
  auto Bb = self_times_transpose(Fb);
  // Deviatoric Cauchy stress
  auto const TRBb = trace(Bb) / 3.0;
  for (int i = 0; i < 3; ++i) Bb(i, i) -= TRBb;
  auto T = hpc::symmetric_stress<double>(EG * Bb);
  // Pressure response
  auto const PR = 2.0 / D1 * (jac - 1.0);
  for (int i = 0; i < 3; ++i) T(i, i) += PR;
  return T;
}

HPC_NOINLINE inline
ErrorCode
update(Properties const props, hpc::deformation_gradient<double> const F,
    double const dtime, double const temp, hpc::symmetric_stress<double>& T,
    hpc::deformation_gradient<double>& Fp, double& ep, double& epdot, double& dp, int& localized)
{
  auto const jac = determinant(F);

  // Determine the stress predictor.
  hpc::symmetric_stress<double> Te;
  auto const Fe = F * 1.;  //  inverse(Fp);
  ErrorCode err_c = ErrorCode::NOT_SET;
  if (props.elastic == Elastic::LINEAR_ELASTIC) {
    Te = linear_elastic_stress(props, Fe);
  }
  else if (props.elastic == Elastic::NEO_HOOKEAN) {
    Te = hyper_elastic_stress(props, Fe, jac);
  }

  // check yield and perform radial return (if applicable)
  auto flag = StateFlag::TRIAL;
  err_c = radial_return(props, Te, F, temp, dtime, T, Fp, ep, epdot, dp, flag);
  if (err_c != ErrorCode::SUCCESS) {
    return err_c;
  }

  bool is_localized = false;
  auto p = -trace(T) / 3.;
  auto const I = hpc::symmetric_stress<double>::identity();
  if (props.damage != Damage::NONE) {
    // If the particle has already failed, apply various erosion algorithms
    if (localized > 0) {
      if (props.allow_no_tension) {
        if (p < 0.0) {
          T = 0.0 * I;
        } else {
          T = -p * I;
        }
      }
      else if (props.allow_no_shear) {
        T = -p * I;
      }
      else if (props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }

    // Update damage and check modified TEPLA rule
    dp = scalar_damage(props, T, dp, temp, ep, epdot, dtime);
    double const por = 0.0;
    double const por_crit = 1.0;
    double const por_ratio = por / por_crit;
    double const D0dpDC = (props.D0 + dp) / props.DC;
    auto const tepla = por_ratio * por_ratio + D0dpDC * D0dpDC;
    if (tepla > 1.0) {
      is_localized = true;
    }
  }

  if (is_localized) {
    // If the localized material point fails again, set the stress to zero
    if (localized > 0) {
      dp = 0.0;
      T = 0.0 * I;
    } else {
      // set the particle localization flag to true
      localized = 1;
      dp = 0.0;
      // Apply various erosion algorithms
      if (props.allow_no_tension) {
        if (p < 0.0) {
          T = 0.0 * I;
        } else {
          T = -p * I;
        }
      }
      else if (props.allow_no_shear) {
        T = -p * I;
      }
      else if (props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }
  }
  return ErrorCode::SUCCESS;
}

}  // namespace hyper_ep
}  // namespace lgr

#endif  // LGR_HYPER_EP_MODEL_RT_HPP
