#ifndef LGR_MODELS_HPP
#define LGR_MODELS_HPP

#include <lgr_element_types.hpp>
#include <lgr_factories.hpp>
#include <lgr_model.hpp>

namespace lgr {

struct Models {
  Simulation& sim;
  std::vector<std::unique_ptr<ModelBase>> models;
  Models(Simulation& sim_in);
  void setup_material_models_and_modifiers(Omega_h::InputMap& pl);
  void setup_field_updates(Omega_h::InputMap& pl);

  void learn_disc();
  void after_configuration();
  void before_field_update();
  void at_field_update();
  void after_field_update();
  void before_material_model();
  void at_material_model();
  void after_material_model();
  void before_secondaries();
  void at_secondaries();
  void after_secondaries();
  void after_correction();

  void add(ModelBase* new_model);
  void run(std::string const& name);
};

template <class Elem>
ModelFactories get_builtin_material_model_factories();
template <class Elem>
ModelFactories get_builtin_modifier_factories();
template <class Elem>
ModelFactories get_builtin_field_update_factories();

#define LGR_EXPL_INST(Elem)                                                    \
  extern template ModelFactories get_builtin_material_model_factories<Elem>(); \
  extern template ModelFactories get_builtin_modifier_factories<Elem>();       \
  extern template ModelFactories get_builtin_field_update_factories<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
