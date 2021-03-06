#include <Omega_h_array_ops.hpp>
#include <Omega_h_profile.hpp>
#include <iostream>
#include <lgr_simulation.hpp>

namespace lgr {

int Simulation::dim() { return disc.dim(); }

int Simulation::nodes() { return disc.count(NODES); }

int Simulation::elems() { return disc.count(ELEMS); }

int Simulation::points() {
  return disc.count(ELEMS) * disc.points_per_ent(ELEMS);
}

Omega_h::LOs Simulation::elems_to_nodes() { return disc.ents_to_nodes(ELEMS); }

Omega_h::Adj Simulation::nodes_to_elems() { return disc.nodes_to_ents(ELEMS); }

void Simulation::finalize_definitions() {
  fields.finalize_definitions(supports);
}

Omega_h::Read<double> Simulation::get(FieldIndex fi) { return fields.get(fi); }

Omega_h::Write<double> Simulation::set(FieldIndex fi) { return fields.set(fi); }

Omega_h::Write<double> Simulation::getset(FieldIndex fi) {
  return fields.getset(fi);
}

MappedRead Simulation::get(FieldIndex fi, Subset* subset) {
  MappedRead mr;
  mr.data = fields.get(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mr.mapping = bridge->mapping;
  return mr;
}

MappedWrite Simulation::set(FieldIndex fi, Subset* subset) {
  MappedWrite mw;
  mw.data = fields.set(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mw.mapping = bridge->mapping;
  return mw;
}

MappedWrite Simulation::getset(FieldIndex fi, Subset* subset) {
  MappedWrite mw;
  mw.data = fields.getset(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mw.mapping = bridge->mapping;
  return mw;
}

template <class Elem>
MappedPointRead<Elem> Simulation::points_get(FieldIndex fi, Subset* subset) {
  MappedPointRead<Elem> mr;
  mr.data = fields.get(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mr.mapping = bridge->mapping;
  return mr;
}

template <class Elem>
MappedPointWrite<Elem> Simulation::points_set(FieldIndex fi, Subset* subset) {
  MappedPointWrite<Elem> mw;
  mw.data = fields.set(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mw.mapping = bridge->mapping;
  return mw;
}

template <class Elem>
MappedPointWrite<Elem> Simulation::points_getset(
    FieldIndex fi, Subset* subset) {
  MappedPointWrite<Elem> mw;
  mw.data = fields.getset(fi);
  auto bridge = subsets.get_bridge(subset, fields[fi].support->subset);
  mw.mapping = bridge->mapping;
  return mw;
}

void Simulation::del(FieldIndex fi) { fields.del(fi); }

Simulation::Simulation(Omega_h::CommPtr comm_in, Setups const& setups_in, Factories&& factories_in)
    : comm(comm_in),
      factories(std::move(factories_in)),
      setups(setups_in),
      input_variables(*this),
      disc(),
      subsets(disc),
      supports(subsets),
      models(*this),
      scalars(*this),
      responses(*this),
      adapter(*this),
      flooder(*this) {}

void Simulation::setup(Omega_h::InputMap& pl) {
  OMEGA_H_CHECK(pl.used);
  OMEGA_H_TIME_FUNCTION;
  start_cpu_time_point = Omega_h::now();
  input_variables.setup(pl.get_map("input variables"));
  // set up constants
  cpu_time = get_double(pl, "start CPU time", "0.0");
  time = get_double(pl, "start time", "0.0");
  prev_time = time;
  auto const dbl_max = std::to_string(std::numeric_limits<double>::max());
  auto const int_max = std::to_string(std::numeric_limits<int>::max());
  end_time = get_double(pl, "end time", dbl_max.c_str());
  dt = 0.0;
  prev_dt = 0.0;
  max_dt = get_double(pl, "max dt", dbl_max.c_str());
  min_dt = get_double(pl, "min dt", "0.0");
  cfl = get_double(pl, "CFL", "0.9");
  step = pl.get<int>("start step", "0");
  end_step = pl.get<int>("end step", int_max.c_str());
  // done setting up constants
  // set up mesh
  disc.setup(comm, pl.get_map("mesh"),input_variables.env);
  // done setting up mesh
  // start defining fields
  fields.setup(pl);
  auto& everywhere = disc.covering_class_names();
  ClassNames nowhere;
  position = fields.define("x", "position", dim(), NODES, false, everywhere);
  // fields[position].remap_type = RemapType::NODAL; // position is special
  velocity = fields.define("v", "velocity", dim(), NODES, false, everywhere);
  fields[velocity].remap_type = RemapType::NODAL;
  fields[velocity].default_value = "vector(0.0)";
  acceleration =
      fields.define("a", "acceleration", dim(), NODES, false, everywhere);
  fields[acceleration].remap_type = RemapType::NODAL;
  force = fields.define("f", "force", dim(), NODES, false, everywhere);
  stress = fields.define(
      "sigma", "stress", Omega_h::symm_ncomps(3), ELEMS, true, everywhere);
  gradient = fields.define("grad", "gradient",
      disc.nodes_per_ent(ELEMS) * dim(), ELEMS, true, everywhere);
  fields[gradient].remap_type = RemapType::SHAPE;
  weight = fields.define("w", "weight", 1, ELEMS, true, everywhere);
  fields[weight].remap_type = RemapType::SHAPE;
  time_step_length =
      fields.define("h", "time step length", 1, ELEMS, false, everywhere);
  fields[time_step_length].remap_type = RemapType::SHAPE;
  viscosity_length =
      fields.define("h_visc", "viscosity length", 1, ELEMS, false, everywhere);
  fields[viscosity_length].remap_type = RemapType::SHAPE;
  wave_speed = fields.define("c", "wave speed", 1, ELEMS, true, everywhere);
  density = fields.define("rho", "density", 1, ELEMS, true, everywhere);
  fields[density].remap_type = RemapType::PER_UNIT_VOLUME;
  nodal_mass = fields.define("m", "nodal mass", 1, NODES, false, everywhere);
  point_time_step =
      fields.define("dt", "time step", 1, ELEMS, true, everywhere);
  traction = fields.define("tau", "traction", 1, SIDES, true, nowhere);
  // done defining fields
  models.setup_material_models_and_modifiers(pl);
  flooder.setup(pl);
  models.setup_field_updates(pl);
  finalize_definitions();
  // setup conditions
  fields.setup_conditions(*this, pl.get_map("conditions"));
  fields.setup_common_defaults(pl.get_map("common fields"));
  fields.setup_default_conditions(*this, time);
  // done setting up conditions
  // setup circuit
  circuit.Setup(input_variables.env,pl);
  // done settting up circuit
  // set coordinates
  auto const field_x = fields.set(position);
  auto const mesh_x = disc.get_node_coords();
  Omega_h::copy_into(mesh_x, field_x);
  // done setting coordinates
  scalars.setup(pl.get_map("scalars"));
  no_output = pl.get<bool>("no output", "false");
  for (auto& setup : setups.responses) {
    setup(*this, pl);
  }
  responses.setup(pl.get_list("responses"));
  adapter.setup(pl);
  // echo parameters
  if (pl.get<bool>("echo parameters", "false")) {
    Omega_h::echo_input(std::cout, pl);
  }
  OMEGA_H_CHECK(pl.used);
  Omega_h::check_unused(pl);
}

void apply_conditions(Simulation& sim, FieldIndex fi) {
  auto& f = sim.fields[fi];
  f.apply_conditions(
      sim.prev_time, sim.time, sim.get(sim.position), sim.fields);
}

void apply_conditions(Simulation& sim) {
  for (auto& f : sim.fields.storage) {
    f->apply_conditions(
        sim.prev_time, sim.time, sim.get(sim.position), sim.fields);
  }
}

void update_time(Simulation& sim) {
  sim.prev_time = sim.time;
  sim.prev_dt = sim.dt;
  auto points_to_dt = sim.get(sim.point_time_step);
  auto min_point_dt = Omega_h::get_min(sim.comm, points_to_dt);
  sim.dt = min_point_dt * sim.cfl;
  sim.dt = Omega_h::min2(sim.dt, sim.max_dt);
  sim.time = sim.prev_time + sim.dt;
  auto next_event = sim.fields.next_event(sim.prev_time);
  next_event =
      Omega_h::min2(next_event, sim.responses.next_event(sim.prev_time));
  next_event = Omega_h::min2(next_event, sim.end_time);
  if (next_event < sim.time) {
    sim.time = next_event;
    sim.dt = sim.time - sim.prev_time;
  }
  if (sim.dt < sim.min_dt) {
    Omega_h_fail("Simulation dt %g went below user-specified minimum dt %g\n",
        sim.dt, sim.min_dt);
  }
}

void update_cpu_time(Simulation& sim) {
  sim.prev_cpu_time = sim.cpu_time;
  auto now = Omega_h::now();
  sim.cpu_time = now - sim.start_cpu_time_point;
}

double Simulation::get_double(
    Omega_h::InputMap& pl, const char* name, const char* default_expr) {
  return input_variables.get_double(pl, name, default_expr);
}

int Simulation::get_int(
    Omega_h::InputMap& pl, const char* name, const char* default_expr) {
  return input_variables.get_int(pl, name, default_expr);
}

template <class Elem>
void Simulation::set_elem() {
  elem_name = Elem::name();
  disc.set_elem<Elem>();
  supports.set_elem<Elem>();
}

#define LGR_EXPL_INST(Elem) template void Simulation::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

#define LGR_EXPL_INST(Elem)                                                    \
  template MappedPointRead<Elem> Simulation::points_get<Elem>(                 \
      FieldIndex, Subset*);                                                    \
  template MappedPointWrite<Elem> Simulation::points_set<Elem>(                \
      FieldIndex, Subset*);                                                    \
  template MappedPointWrite<Elem> Simulation::points_getset<Elem>(             \
      FieldIndex, Subset*);
LGR_EXPL_INST_ELEMS_AND_SIDES
#undef LGR_EXPL_INST

}  // namespace lgr
