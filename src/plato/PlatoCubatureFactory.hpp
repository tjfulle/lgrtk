/*
 * PlatoCubatureFactory.hpp
 *
 *  Created on: May 2, 2019
 */

#ifndef PLATOCUBATUREFACTORY_HPP_
#define PLATOCUBATUREFACTORY_HPP_

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Teuchos_ParameterList.hpp>

#ifdef PLATO_GEOMETRY
#include "plato/CogentCubature.hpp"
#endif
#include "plato/LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/**********************************************************************************/
template<int SpatialDim>
class CubatureFactory
{
/**********************************************************************************/
public:
    std::shared_ptr<Plato::CubatureRule<SpatialDim>> 
    create(Omega_h::Mesh& aMesh, Teuchos::ParameterList& aProblemParams)
    {

        bool hasCubatureSpec = aProblemParams.isSublist("Cubature");

        if(hasCubatureSpec)
        {
            auto tCubatureSpec = aProblemParams.sublist("Cubature");
            auto tCubatureType = tCubatureSpec.get<std::string>("Type");
            auto tTypeSpec = tCubatureSpec.sublist(tCubatureType);
            if(tCubatureType == "Gauss")
            {
                return std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpatialDim>>(aMesh);
            } else
            if(tCubatureType == "Cogent")
            {
#ifdef PLATO_GEOMETRY
                return std::make_shared<Plato::CogentCubature<SpatialDim>>(aMesh, tTypeSpec);
#else
                std::ostringstream tErrorMessage;
                tErrorMessage << " Fatal error -- Cogent cubature not compiled.  " << tCubatureType << std::endl;
                throw std::runtime_error(tErrorMessage.str().c_str());
#endif
            } else
            {
                std::ostringstream tErrorMessage;
                tErrorMessage << " Fatal error -- Unknown cubature type requested: " << tCubatureType << std::endl;
                throw std::runtime_error(tErrorMessage.str().c_str());
            }
        }
        else
        {
            return std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpatialDim>>(aMesh);
        }
    }

    std::shared_ptr<Plato::CubatureRule<SpatialDim>> 
    create(Omega_h::Mesh& aMesh)
    {
        return std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpatialDim>>(aMesh);
    }
};
// class CubatureFactory

} // namespace Plato

#endif /* PLATOCUBATUREFACTORY_HPP_ */
