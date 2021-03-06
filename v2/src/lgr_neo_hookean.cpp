#include <lgr_for.hpp>
#include <lgr_neo_hookean.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

OMEGA_H_INLINE void neo_hookean_update(double bulk_modulus,
    double shear_modulus, double density, Tensor<3> F, Tensor<3>& stress,
    double& wave_speed, double& tangent_bulk_modulus) {
  OMEGA_H_CHECK(density > 0.0);
  auto const J = Omega_h::determinant(F);
  OMEGA_H_CHECK(J > 0.0);
  auto const Jinv = 1.0 / J;
  auto const half_bulk_modulus = (1.0 / 2.0) * bulk_modulus;
  auto const volumetric_stress = half_bulk_modulus * (J - Jinv);
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const isotropic_stress = volumetric_stress * I;
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Jm53 = Jm23 * Jm23 * Jm13;
  auto const B = F * transpose(F);
  auto const devB = Omega_h::deviator(B);
  auto const deviatoric_stress = shear_modulus * Jm53 * devB;
  stress = isotropic_stress + deviatoric_stress;
  tangent_bulk_modulus = half_bulk_modulus * (J + Jinv);
  auto const plane_wave_modulus =
      tangent_bulk_modulus + (4.0 / 3.0) * shear_modulus;
  OMEGA_H_CHECK(plane_wave_modulus > 0.0);
  wave_speed = std::sqrt(plane_wave_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

template <class Elem>
struct NeoHookean : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex effective_bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  NeoHookean(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->bulk_modulus = this->point_define(
        "kappa", "bulk modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    this->shear_modulus = this->point_define(
        "mu", "shear modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient = this->point_define(
        "F", "deformation gradient", square(dim), RemapType::POLAR, pl, "I");
    this->effective_bulk_modulus = this->point_define(
        "kappa_tilde", "effective bulk modulus", 1, RemapType::NONE, pl, "");
  }
  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "neo-Hookean"; }
  void at_material_model() override final {
    auto points_to_kappa = this->points_get(this->bulk_modulus);
    auto points_to_nu = this->points_get(this->shear_modulus);
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_F = this->points_get(this->deformation_gradient);
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_wave_speed = this->points_set(this->sim.wave_speed);
    auto points_to_kappa_tilde = this->points_set(this->effective_bulk_modulus);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto F_small = getfull<Elem>(points_to_F, point);
      auto kappa = points_to_kappa[point];
      auto nu = points_to_nu[point];
      auto rho = points_to_rho[point];
      auto F = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i) {
        for (int j = 0; j < Elem::dim; ++j) {
          F(i, j) = F_small(i, j);
        }
      }
      Tensor<3> sigma;
      double c, kappa_tilde;
      neo_hookean_update(kappa, nu, rho, F, sigma, c, kappa_tilde);
      setstress(points_to_stress, point, sigma);
      points_to_wave_speed[point] = c;
      points_to_kappa_tilde[point] = kappa_tilde;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

void setup_neo_hookean(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("material models");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "neo-Hookean") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new NeoHookean<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
