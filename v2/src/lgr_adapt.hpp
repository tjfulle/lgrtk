#ifndef LGR_ADAPT_HPP
#define LGR_ADAPT_HPP

#include <Omega_h_input.hpp>
#include <lgr_remap.hpp>

namespace lgr {

struct Simulation;

struct Adapter {
  Simulation& sim;
  Omega_h::AdaptOpts opts;
  std::shared_ptr<RemapBase> remap;
  bool should_adapt;
  double trigger_quality;
  double trigger_length_ratio;
  double minimum_length;
  double gradation_rate;
  bool should_coarsen_with_expansion;
  Adapter(Simulation& sim);
  void setup(Omega_h::InputMap& pl);
  bool adapt();
  void coarsen_metric_with_expansion();
  double old_quality;
  double old_length;
};

}  // namespace lgr

#endif
