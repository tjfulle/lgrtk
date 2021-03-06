#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include "plato/PlatoStaticsTypes.hpp"
#include "plato/ScalarFunctionIncBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/******************************************************************************//**
 * @brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionIncBaseFactory
{
public:
    /******************************************************************************//**
     * @brief Constructor
     **********************************************************************************/
    ScalarFunctionIncBaseFactory () {}

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    ~ScalarFunctionIncBaseFactory() {}

    /******************************************************************************//**
     * @brief Create method
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::ScalarFunctionIncBase> 
    create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName);
};
// class ScalarFunctionIncBaseFactory


}
// namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "StabilizedMechanics.hpp"
#include "Thermomechanics.hpp"
#include "StabilizedThermomechanics.hpp"

#ifdef PLATO_1D
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<1>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<1>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedMechanics<1>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif

#ifdef PLATO_2D
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<2>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<2>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedMechanics<2>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif

#ifdef PLATO_3D
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermal<3>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::Thermomechanics<3>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedMechanics<3>>;
extern template class Plato::ScalarFunctionIncBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
