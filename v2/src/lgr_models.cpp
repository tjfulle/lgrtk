#include <Omega_h_profile.hpp>
#include <lgr_joule_heating.hpp>
#include <lgr_j2_plasticity.hpp>
#include <lgr_mie_gruneisen.hpp>
#include <lgr_models.hpp>
#include <lgr_nodal_pressure.hpp>
#include <lgr_pressure.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>
#include <lgr_stvenant_kirchhoff.hpp>

namespace lgr {

Models::Models(Simulation& sim_in) : sim(sim_in) {}

void Models::add(ModelBase* new_model) {
  std::unique_ptr<ModelBase> uptr(new_model);
  models.push_back(std::move(uptr));
}

void Models::setup_material_models_and_modifiers(Omega_h::InputMap& pl) {
  for (auto& setup : sim.setups.material_models) {
    setup(sim, pl);
  }
  ::lgr::setup(sim.factories.material_model_factories, sim,
      pl.get_list("material models"), models, "material model");
  for (auto& model_ptr : models) {
    OMEGA_H_CHECK((model_ptr->exec_stages() & AT_MATERIAL_MODEL) != 0);
  }
  for (auto& setup : sim.setups.modifiers) {
    setup(sim, pl);
  }
  ::lgr::setup(sim.factories.modifier_factories, sim, pl.get_list("modifiers"),
      models, "modifier");
}

void Models::setup_field_updates(Omega_h::InputMap& pl) {
  for (auto& setup : sim.setups.field_updates) {
    setup(sim, pl);
  }
  auto const& factories = sim.factories.field_update_factories;
  Omega_h::InputMap dummy_pl;
  // this can't be a range-based for loop because some field update
  // models create fields which alters the sim.fields.storage vector
  // which invalidates most iterators to it including the one used by
  // the range based for loop
  for (std::size_t i = 0; i < sim.fields.storage.size(); ++i) {
    auto& f_ptr = sim.fields.storage[i];
    auto& name = f_ptr->long_name;
    auto it = factories.find(name);
    if (it == factories.end()) continue;
    auto& factory = it->second;
    auto ptr = factory(sim, name, dummy_pl);
    std::unique_ptr<ModelBase> unique_ptr(ptr);
    models.push_back(std::move(unique_ptr));
  }
}

void Models::learn_disc() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    model->learn_disc();
  }
}

#define LGR_STAGE_DEF(lowercase, uppercase)                                    \
  void Models::lowercase() {                                                   \
    OMEGA_H_TIME_FUNCTION;                                                     \
    for (auto& model : models) {                                               \
      if ((model->exec_stages() & uppercase) != 0) {                           \
        Scope scope{sim, model->name()};                                       \
        model->lowercase();                                                    \
      }                                                                        \
    }                                                                          \
  }
LGR_STAGE_DEF(after_configuration, AFTER_CONFIGURATION)
LGR_STAGE_DEF(before_field_update, BEFORE_FIELD_UPDATE)
LGR_STAGE_DEF(at_field_update, AT_FIELD_UPDATE)
LGR_STAGE_DEF(after_field_update, AFTER_FIELD_UPDATE)
LGR_STAGE_DEF(before_material_model, BEFORE_MATERIAL_MODEL)
LGR_STAGE_DEF(at_material_model, AT_MATERIAL_MODEL)
LGR_STAGE_DEF(after_material_model, AFTER_MATERIAL_MODEL)
LGR_STAGE_DEF(before_secondaries, BEFORE_SECONDARIES)
LGR_STAGE_DEF(at_secondaries, AT_SECONDARIES)
LGR_STAGE_DEF(after_secondaries, AFTER_SECONDARIES)
LGR_STAGE_DEF(after_correction, AFTER_CORRECTION)
#undef LGR_STAGE_DEF

void Models::run(std::string const& name) {
  for (auto& model : models) {
    if (name == model->name()) {
      Scope scope{sim, name.c_str()};
    //model->run();
    }
  }
}

template <class Elem>
ModelFactories get_builtin_material_model_factories() {
  ModelFactories out;
  out["J2 plasticity"] = j2_plasticity_factory<Elem>;
  out["Mie-Gruneisen"] = mie_gruneisen_factory<Elem>;
  out["StVenant-Kirchhoff"] = stvenant_kirchhoff_factory<Elem>;
  return out;
}

template <class Elem>
ModelFactories get_builtin_modifier_factories() {
  ModelFactories out;
  out["nodal pressure"] = nodal_pressure_factory<Elem>;
  out["compute pressure"] = pressure_factory;
  return out;
}

template <class Elem>
ModelFactories get_builtin_field_update_factories() {
  ModelFactories out;
  return out;
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelFactories get_builtin_material_model_factories<Elem>();        \
  template ModelFactories get_builtin_modifier_factories<Elem>();              \
  template ModelFactories get_builtin_field_update_factories<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
