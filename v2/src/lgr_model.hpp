#ifndef LGR_MODEL_HPP
#define LGR_MODEL_HPP

#include <Omega_h_input.hpp>
#include <lgr_class_names.hpp>
#include <lgr_element_types.hpp>
#include <lgr_field_access.hpp>
#include <lgr_field_index.hpp>
#include <lgr_remap_type.hpp>
#include <Omega_h_expr.hpp>

namespace lgr {

struct Simulation;
struct Support;

enum ExecStage : std::uint64_t {
  AFTER_CONFIGURATION = std::uint64_t(1) << 0,
  BEFORE_FIELD_UPDATE = std::uint64_t(1) << 1,
  AT_FIELD_UPDATE = std::uint64_t(1) << 2,
  AFTER_FIELD_UPDATE = std::uint64_t(1) << 3,
  BEFORE_MATERIAL_MODEL = std::uint64_t(1) << 4,
  AT_MATERIAL_MODEL = std::uint64_t(1) << 5,
  AFTER_MATERIAL_MODEL = std::uint64_t(1) << 6,
  BEFORE_SECONDARIES = std::uint64_t(1) << 7,
  AT_SECONDARIES = std::uint64_t(1) << 8,
  AFTER_SECONDARIES = std::uint64_t(1) << 9,
  AFTER_CORRECTION = std::uint64_t(1) << 10,
};

struct ModelBase {
  Simulation& sim;
  Support* elem_support;
  Support* point_support;
  virtual ~ModelBase() = default;
  virtual void out_of_line_virtual_function();
  ModelBase(Simulation& sim_in, ClassNames const& class_names);
  ModelBase(Simulation& sim_in, Omega_h::InputMap& pl);
  virtual std::uint64_t exec_stages() = 0;
  virtual char const* name() = 0;
  FieldIndex point_define(
      std::string const& short_name, std::string const& long_name, int ncomps);
  FieldIndex point_define(std::string const& short_name,
      std::string const& long_name, int ncomps,
      std::string const& default_value);
  FieldIndex point_define(std::string const& short_name,
      std::string const& long_name, int ncomps, RemapType tt,
      std::string const& default_value);
  FieldIndex point_define(std::string const& short_name,
      std::string const& long_name, int ncomps, RemapType tt,
      Omega_h::InputMap& pl, std::string const& default_value);
  FieldIndex elem_define(
      std::string const& short_name, std::string const& long_name, int ncomps);
  FieldIndex elem_define(std::string const& short_name,
      std::string const& long_name, int ncomps,
      std::string const& default_value);
  FieldIndex elem_define(std::string const& short_name,
      std::string const& long_name, int ncomps, RemapType tt,
      std::string const& default_value);
  MappedElemsToNodes get_elems_to_nodes();
  int points();
  int elems();
  MappedRead elems_get(FieldIndex fi);
  MappedWrite elems_set(FieldIndex fi);
  MappedWrite elems_getset(FieldIndex fi);
  virtual void learn_disc();
  virtual void after_configuration();
  virtual void before_field_update();
  virtual void at_field_update();
  virtual void after_field_update();
  virtual void before_material_model();
  virtual void at_material_model();
  virtual void after_material_model();
  virtual void before_secondaries();
  virtual void at_secondaries();
  virtual void after_secondaries();
  virtual void after_correction();
  double get_double(
      Omega_h::InputMap& pl, const char* name, const char* default_expr);
};

template <class Elem>
struct Model : public ModelBase {
  Model(Simulation&, Omega_h::InputMap&);
  Model(Simulation&, ClassNames const&);
  MappedPointRead<Elem> points_get(FieldIndex fi);
  MappedPointWrite<Elem> points_set(FieldIndex fi);
  MappedPointWrite<Elem> points_getset(FieldIndex fi);
  MappedPointRead<Elem> elems_get(FieldIndex fi);
  MappedPointWrite<Elem> elems_set(FieldIndex fi);
  MappedPointWrite<Elem> elems_getset(FieldIndex fi);
};

#ifdef _MSC_VER
#define LGR_EXPL_INST(Elem) extern template struct Model<Elem>; \
  extern template MappedPointRead<Elem> Model<Elem>::elems_get(FieldIndex);
#else
#define LGR_EXPL_INST(Elem) extern template struct Model<Elem>;
#endif
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
