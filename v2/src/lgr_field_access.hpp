#ifndef LGR_FIELD_ACCESS_HPP
#define LGR_FIELD_ACCESS_HPP

#include <Omega_h_adj.hpp>
#include <Omega_h_scalar.hpp>
#include <lgr_mapping.hpp>
#include <lgr_math.hpp>

namespace lgr {

using Omega_h::divide_no_remainder;

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(Omega_h::Read<double> const& a, int i) {
  return Omega_h::get_vector<Elem::dim>(a, i);
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(
    Omega_h::Write<double> const& a, int i) {
  return Omega_h::get_vector<Elem::dim>(a, i);
}

template <class Elem>
OMEGA_H_DEVICE void setvec(
    Omega_h::Write<double> const& a, int i, Omega_h::Vector<Elem::dim> v) {
  Omega_h::set_vector(a, i, v);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    Omega_h::Read<double> const& a, int i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(a, i);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    Omega_h::Write<double> const& a, int i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(a, i);
}

template <class Elem>
OMEGA_H_DEVICE void setfull(
    Omega_h::Write<double> const& a, int i, Tensor<Elem::dim> v) {
  Omega_h::set_matrix<Elem::dim, Elem::dim>(a, i, v);
}

OMEGA_H_DEVICE Tensor<3> getstress(Omega_h::Read<double> const& a, int i) {
  return Omega_h::get_symm<3>(a, i);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<3> getstress(Omega_h::Write<double> const& a, int i) {
  return Omega_h::get_symm<3>(a, i);
}

OMEGA_H_DEVICE void setstress(
    Omega_h::Write<double> const& a, int const i, Tensor<3> const v) {
  Omega_h::set_symm<3>(a, i, v);
}

template <class Elem>
OMEGA_H_DEVICE Omega_h::Few<int, Elem::nodes> getnodes(
    Omega_h::LOs const& elems_to_nodes, int elem) {
  return Omega_h::gather_down<Elem::nodes>(elems_to_nodes, elem);
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::nodes> getscals(Omega_h::Read<double> const& data,
    Omega_h::Few<int, Elem::nodes> cell_nodes) {
  return Omega_h::gather_scalars<Elem::nodes>(data, cell_nodes);
}

template <class Elem>
OMEGA_H_DEVICE Matrix<Elem::dim, Elem::nodes> getvecs(
    Omega_h::Read<double> const& data,
    Omega_h::Few<int, Elem::nodes> cell_nodes) {
  return Omega_h::gather_vectors<Elem::nodes, Elem::dim>(data, cell_nodes);
}

template <class Elem>
OMEGA_H_DEVICE Matrix<Elem::dim, Elem::nodes> getgrads(
    Omega_h::Read<double> const& gradients, int point) {
  return transpose(
      Omega_h::get_matrix<Elem::nodes, Elem::dim>(gradients, point));
}

template <class Elem>
OMEGA_H_DEVICE void setgrads(Omega_h::Write<double> const& gradients, int point,
    Matrix<Elem::dim, Elem::nodes> point_grads) {
  return Omega_h::set_matrix<Elem::nodes, Elem::dim>(
      gradients, point, transpose(point_grads));
}

/* \frac{\partial u}{\partial X_i} =
   \sum_j \frac{\partial u}{\partial \xi_j}\frac{\partial \xi_j}{\partial x_i}
 */
template <class Elem>
OMEGA_H_INLINE Vector<Elem::dim> grad(
    Matrix<Elem::dim, Elem::nodes> const basis_grads,
    Vector<Elem::nodes> const node_scals) {
  return basis_grads * node_scals;
}

/* \frac{\partial u_i}{\partial X_j} =
   \sum_k \frac{\partial u_i}{\partial \xi_k}\frac{\partial \xi_k}{\partial x_j}
 */
template <class Elem>
OMEGA_H_INLINE Tensor<Elem::dim> grad(
    Matrix<Elem::dim, Elem::nodes> const basis_grads,
    Matrix<Elem::dim, Elem::nodes> const node_vecs) {
  return node_vecs * transpose(basis_grads);
}

template <class Elem>
OMEGA_H_DEVICE Omega_h::Vector<Elem::points> getweights(
    Omega_h::Read<double> const& a, int elem) {
  return Omega_h::get_vector<Elem::points>(a, elem);
}

struct MappedWrite {
  Mapping mapping;
  Omega_h::Write<double> data;
  OMEGA_H_DEVICE double& operator[](int const i) const {
    return data[mapping[i]];
  }
};

struct MappedRead {
  Mapping mapping;
  Omega_h::Read<double> data;
  OMEGA_H_DEVICE double const& operator[](int const i) const {
    return data[mapping[i]];
  }
};

template <class Elem>
OMEGA_H_DEVICE int map_point(Mapping const& mapping, int const i) {
  auto const elem = i / Elem::points;
  auto const elem_pt = i % Elem::points;
  return mapping[elem] * Elem::points + elem_pt;
}

template <class Elem>
struct MappedPointWrite {
  Mapping mapping;
  Omega_h::Write<double> data;
  OMEGA_H_DEVICE double& operator[](int const i) const {
    return data[map_point<Elem>(mapping, i)];
  }
};

template <class Elem>
struct MappedPointRead {
  Mapping mapping;
  Omega_h::Read<double> data;
  OMEGA_H_DEVICE double const& operator[](int const i) const {
    return data[map_point<Elem>(mapping, i)];
  }
};

struct MappedElemsToNodes {
  Mapping mapping;
  Omega_h::LOs data;
};

template <class Elem>
inline int count_elements(MappedElemsToNodes const& a) {
  if (a.mapping.is_identity)
    return divide_no_remainder(a.data.size(), Elem::nodes);
  return a.mapping.things.size();
}

template <class Elem>
OMEGA_H_DEVICE Omega_h::Few<int, Elem::nodes> getnodes(
    MappedElemsToNodes const& a, int elem) {
  return Omega_h::gather_verts<Elem::nodes>(a.data, a.mapping[elem]);
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(MappedRead const& a, int const i) {
  return Omega_h::get_vector<Elem::dim>(a.data, a.mapping[i]);
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(MappedWrite const& a, int const i) {
  return Omega_h::get_vector<Elem::dim>(a.data, a.mapping[i]);
}

template <class Elem>
OMEGA_H_DEVICE void setvec(
    MappedWrite const& a, int const i, Omega_h::Vector<Elem::dim> v) {
  Omega_h::set_vector(a.data, a.mapping[i], v);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    MappedRead const& a, int const i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(a.data, a.mapping[i]);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    MappedWrite const& a, int const i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(a.data, a.mapping[i]);
}

template <class Elem>
OMEGA_H_DEVICE void setfull(
    MappedWrite const& a, int const i, Tensor<Elem::dim> const v) {
  Omega_h::set_matrix<Elem::dim, Elem::dim>(a.data, a.mapping[i], v);
}

OMEGA_H_DEVICE Tensor<3> getstress(MappedRead const& a, int const i) {
  return Omega_h::get_symm<3>(a.data, a.mapping[i]);
}

OMEGA_H_DEVICE Tensor<3> getstress(MappedWrite const& a, int const i) {
  return Omega_h::get_symm<3>(a.data, a.mapping[i]);
}

OMEGA_H_DEVICE void setstress(
    MappedWrite const& a, int const i, Tensor<3> const v) {
  Omega_h::set_symm<3>(a.data, a.mapping[i], v);
}

template <class Elem>
OMEGA_H_DEVICE Matrix<Elem::dim, Elem::nodes> getvecs(
    MappedRead const& a, Omega_h::Few<int, Elem::nodes> cell_nodes) {
  Matrix<Elem::dim, Elem::nodes> out;
  for (int i = 0; i < Elem::nodes; ++i) {
    out[i] = getvec(a, cell_nodes[i]);
  }
  return out;
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(
    MappedPointRead<Elem> const& a, int const i) {
  return Omega_h::get_vector<Elem::dim>(a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE Vector<Elem::dim> getvec(
    MappedPointWrite<Elem> const& a, int const i) {
  return Omega_h::get_vector<Elem::dim>(a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE void setvec(MappedPointWrite<Elem> const& a, int const i,
    Omega_h::Vector<Elem::dim> v) {
  Omega_h::set_vector(a.data, map_point<Elem>(a.mapping, i), v);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    MappedPointRead<Elem> const& a, int const i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(
      a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE Tensor<Elem::dim> getfull(
    MappedPointWrite<Elem> const& a, int const i) {
  return Omega_h::get_matrix<Elem::dim, Elem::dim>(
      a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE void setfull(MappedPointWrite<Elem> const& a, int const i,
    Tensor<Elem::dim> const v) {
  Omega_h::set_matrix<Elem::dim, Elem::dim>(
      a.data, map_point<Elem>(a.mapping, i), v);
}

template <class Elem>
OMEGA_H_DEVICE Tensor<3> getstress(
    MappedPointRead<Elem> const& a, int const i) {
  return Omega_h::get_symm<3>(a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE Tensor<3> getstress(
    MappedPointWrite<Elem> const& a, int const i) {
  return Omega_h::get_symm<3>(a.data, map_point<Elem>(a.mapping, i));
}

template <class Elem>
OMEGA_H_DEVICE void setstress(
    MappedPointWrite<Elem> const& a, int const i, Tensor<3> const v) {
  Omega_h::set_symm<3>(a.data, map_point<Elem>(a.mapping, i), v);
}

template <class Elem>
OMEGA_H_DEVICE Matrix<Elem::dim, Elem::nodes> getgrads(
    MappedPointRead<Elem> const& a, int point) {
  auto mapped_pt = map_point<Elem>(a.mapping, point);
  return transpose(
      Omega_h::get_matrix<Elem::nodes, Elem::dim>(a.data, mapped_pt));
}

template <class Elem>
OMEGA_H_DEVICE void setgrads(MappedPointRead<Elem> const& a, int const point,
    Matrix<Elem::dim, Elem::nodes> const point_grads) {
  auto mapped_pt = map_point<Elem>(a.mapping, point);
  return Omega_h::set_matrix<Elem::nodes, Elem::dim>(
      a.data, mapped_pt, tranpose(point_grads));
}

}  // namespace lgr

#endif
