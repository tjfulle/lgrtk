/*
 * CubatureRule.hpp
 *
 *  Created on: May 2, 2019
 */

#ifndef CUBATURE_RULE_HPP_
#define CUBATURE_RULE_HPP_

#include <string>
#include <Omega_h_mesh.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CubatureRule
/******************************************************************************/
{
public:
    /******************************************************************************/
    CubatureRule() :
    mNumCubPoints(1),
    mCubWeights("Linear Tet4: Cubature Weights View", 1),
    mCubPointsCoords("Linear Tet4: Cubature Coords View", SpaceDim),
    mBasisFunctions("Linear Tet4: Basis Functions View", SpaceDim+1)
    /******************************************************************************/
    {
        this->initialize();
    }
   
    /******************************************************************************/
    CubatureRule(Omega_h::Mesh& aMesh) : 
    mNumCubPoints(1),
    mCubWeights("Linear Tet4: Cubature Weights View", aMesh.nelems()),
    mCubPointsCoords("Linear Tet4: Cubature Coords View", SpaceDim),
    mBasisFunctions("Linear Tet4: Basis Functions View", SpaceDim+1)
    /******************************************************************************/
    {
        this->initialize();
    }

    /******************************************************************************/
    virtual ~CubatureRule(){}
    /******************************************************************************/

    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::ScalarVector getCubWeights() const
    /******************************************************************************/
    {
        return (mCubWeights);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::OrdinalType getNumCubPoints() const
    /******************************************************************************/
    {
        return (mNumCubPoints);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION const Plato::ScalarVector & getBasisFunctions() const
    /******************************************************************************/
    {
        return (mBasisFunctions);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION const Plato::ScalarVector & getCubPointsCoords() const
    /******************************************************************************/
    {
        return (mCubPointsCoords);
    }

private:
    /****************************************************************************//**/
    /*!
     * For /f$dim == 1/f$, we shift the reference cell from the /f$[-1,1]/f$ interval
     * to a /f$[0,1]/f$ interval. which is consistent with our simplex treatment in
     * higher dimensions. Therefore, the coordinates are transformed by /f$x\rightarrow
     * \frac{x + 1.0}{2.0}/f$ and the weights are cut in half.
     *
     ********************************************************************************/
    void initialize()
    {

        // initialize array with cubature points coordinates
        auto tHostCubPointsCoords = Kokkos::create_mirror(mCubPointsCoords);
        if(SpaceDim == static_cast<Plato::OrdinalType>(3))
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(0.25);
            tHostCubPointsCoords(1) = static_cast<Plato::Scalar>(0.25);
            tHostCubPointsCoords(2) = static_cast<Plato::Scalar>(0.25);
        }
        else if(SpaceDim == static_cast<Plato::OrdinalType>(2))
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(1.0/3.0);
            tHostCubPointsCoords(1) = static_cast<Plato::Scalar>(1.0/3.0);
        }
        else
        {
            tHostCubPointsCoords(0) = static_cast<Plato::Scalar>(1.0/2.0);
        }
        Kokkos::deep_copy(mCubPointsCoords, tHostCubPointsCoords);

        // initialize array with basis functions
        auto tHostBasisFunctions = Kokkos::create_mirror(mBasisFunctions);
        if(SpaceDim == static_cast<Plato::OrdinalType>(3))
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0)
                    - tHostCubPointsCoords(1) - tHostCubPointsCoords(2);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
            tHostBasisFunctions(2) = tHostCubPointsCoords(1);
            tHostBasisFunctions(3) = tHostCubPointsCoords(2);
        }
        else if(SpaceDim == static_cast<Plato::OrdinalType>(2))
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0) - tHostCubPointsCoords(1);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
            tHostBasisFunctions(2) = tHostCubPointsCoords(1);
        }
        else
        {
            tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1) - tHostCubPointsCoords(0);
            tHostBasisFunctions(1) = tHostCubPointsCoords(0);
        }

        Kokkos::deep_copy(mBasisFunctions, tHostBasisFunctions);
    }

protected:
    Plato::OrdinalType  mNumCubPoints;
    Plato::ScalarVector mCubWeights;
    Plato::ScalarVector mCubPointsCoords;
    Plato::ScalarVector mBasisFunctions;
};
// class CubatureRule

} // namespace Plato

#endif
