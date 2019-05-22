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
#include "plato/GaussLegendre.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CubatureRuleDegreeOne
/******************************************************************************/
{
    static constexpr Plato::OrdinalType mDegree=1;
  
    using Cubature = Plato::GaussLegendre<Plato::Scalar>;

public:
    /******************************************************************************/
    CubatureRuleDegreeOne() :
      mNumCubPoints(1),
      mCubWeights("Linear Tet4: Cubature Weights View", mNumCubPoints),
      mCubPointsCoords("Linear Tet4: Cubature Coords View", mNumCubPoints, SpaceDim),
      mBasisFunctions("Linear Tet4: Basis Functions View", mNumCubPoints, SpaceDim+1)
    /******************************************************************************/
    {
        this->initialize(Cubature::Rules[SpaceDim-1][mDegree]);
    }
   
    /******************************************************************************/
    CubatureRuleDegreeOne(Omega_h::Mesh& aMesh) :
      mNumCubPoints(aMesh.nelems()),
      mCubWeights("Linear Tet4: Cubature Weights View", mNumCubPoints),
      mCubPointsCoords("Linear Tet4: Cubature Coords View", mNumCubPoints, SpaceDim),
      mBasisFunctions("Linear Tet4: Basis Functions View", mNumCubPoints, SpaceDim+1)
    /******************************************************************************/
    {
        this->initialize(Cubature::Rules[SpaceDim-1][mDegree]);
    }

    /******************************************************************************/
    virtual ~CubatureRuleDegreeOne(){}
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
    void initialize(const Plato::GaussLegendre<Plato::Scalar>::Rule& aGauss)
    {
        // initialize cubature points coordinates and basis functions
        auto tHostCubPointsCoords = Kokkos::create_mirror(mCubPointsCoords);
        auto tHostBasisFunctions  = Kokkos::create_mirror(mBasisFunctions);

        tHostBasisFunctions(0) = static_cast<Plato::Scalar>(1);
        for( int iDim=0; iDim<SpaceDim; ++iDim ){
            tHostCubPointsCoords(iDim) = aGauss.points[0][iDim];
            tHostBasisFunctions(iDim) = tHostCubPointsCoords(iDim);
            tHostBasisFunctions(0) -= tHostCubPointsCoords(iDim);
        }
        Kokkos::deep_copy(mCubPointsCoords, tHostCubPointsCoords);
        Kokkos::deep_copy(mBasisFunctions,  tHostBasisFunctions);
    }

protected:
    Plato::OrdinalType  mNumCubPoints;
    Plato::ScalarVector mCubWeights;
    Plato::ScalarVector mCubPointsCoords;
    Plato::ScalarVector mBasisFunctions;
};


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CubatureRule
/******************************************************************************/
{
  
    //Plato::GaussLegendre<Plato::Scalar> GaussLegendre;
    using Cubature = Plato::GaussLegendre<Plato::Scalar>;

public:
    /******************************************************************************/
    CubatureRule() :
    mNumCubPoints(1),
    mCubWeights("Linear Tet4: Cubature Weights View", mNumCubPoints),
    mCubPointsCoords("Linear Tet4: Cubature Coords View", mNumCubPoints, SpaceDim),
    mBasisFunctions("Linear Tet4: Basis Functions View", mNumCubPoints, SpaceDim+1)
    /******************************************************************************/
    {
        this->initialize(Cubature::Rules[SpaceDim-1][1]);
    }
   
    /******************************************************************************/
    CubatureRule(Omega_h::Mesh& aMesh, int aDegree=1)
    /******************************************************************************/
    {
        if( aDegree >= Cubature::MaxNumDegree )
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << " Fatal error -- Cubature degree " << aDegree << " not implemented." << std::endl;
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto& gauss = Cubature::Rules[SpaceDim-1][aDegree];
        mNumCubPoints = gauss.npoints;
        mCubPointsCoords = Plato::ScalarMultiVector("Linear Tet4: Cubature Coords View", mNumCubPoints, SpaceDim);
        mBasisFunctions  = Plato::ScalarMultiVector("Linear Tet4: Basis Functions View", mNumCubPoints, SpaceDim+1);
        mCubWeights      = Plato::ScalarMultiVector("Linear Tet4: Cubature Weights View", aMesh.nelems(), mNumCubPoints);

        this->initialize(gauss);
    }

    /******************************************************************************/
    virtual ~CubatureRule(){}
    /******************************************************************************/

    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::ScalarMultiVector getCubWeights() const
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
    KOKKOS_INLINE_FUNCTION const Plato::ScalarMultiVector & getBasisFunctions() const
    /******************************************************************************/
    {
        return (mBasisFunctions);
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION const Plato::ScalarMultiVector & getCubPointsCoords() const
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
    void initialize(const Plato::GaussLegendre<Plato::Scalar>::Rule& aGauss)
    {
        // initialize cubature points coordinates and basis functions
        auto tHostCubPointsCoords = Kokkos::create_mirror(mCubPointsCoords);
        auto tHostBasisFunctions  = Kokkos::create_mirror(mBasisFunctions);

        auto tNumGPs = tHostCubPointsCoords.extent(0);
        for( int iGP=0; iGP<tNumGPs; ++iGP ){
            tHostBasisFunctions(iGP,0) = static_cast<Plato::Scalar>(1);
            for( int iDim=0; iDim<SpaceDim; ++iDim ){
                tHostCubPointsCoords(iGP, iDim) = aGauss.points[iGP][iDim];
                tHostBasisFunctions(iGP, iDim) = tHostCubPointsCoords(iGP, iDim);
                tHostBasisFunctions(iGP, 0) -= tHostCubPointsCoords(iGP, iDim);
            }
        }
        Kokkos::deep_copy(mCubPointsCoords, tHostCubPointsCoords);
        Kokkos::deep_copy(mBasisFunctions,  tHostBasisFunctions);
    }

protected:
    Plato::OrdinalType  mNumCubPoints;
    Plato::ScalarMultiVector mCubWeights;
    Plato::ScalarMultiVector mCubPointsCoords;
    Plato::ScalarMultiVector mBasisFunctions;
};
// class CubatureRule

} // namespace Plato

#endif
