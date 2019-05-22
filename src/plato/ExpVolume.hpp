/*
 * ExpVolume.hpp
 *
 *  Created on: Apr 29, 2018
 */

#ifndef PLATO_EXPVOLUME_HPP_
#define PLATO_EXPVOLUME_HPP_

#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "ImplicitFunctors.hpp"

#include "plato/ApplyProjection.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/PlatoCubatureFactory.hpp"
#include "plato/SimplexStructuralDynamics.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFuncType, typename ProjectionFuncType>
class ExpVolume :
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
private:
    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

private:
    PenaltyFuncType mPenaltyFunction;
    ProjectionFuncType mProjectionFunction;
    Plato::ApplyProjection<ProjectionFuncType> mApplyProjection;

    std::shared_ptr<Plato::CubatureRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;


public:
    /**************************************************************************/
    explicit ExpVolume(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap aDataMap, 
                       Teuchos::ParameterList & aPenaltyParams) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Experimental Volume"),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyProjection(mProjectionFunction)
    /**************************************************************************/
    {
         Plato::DegreeOneCubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
         mCubatureRule = tCubatureFactory.create(aMesh);
    }

    /**************************************************************************/
    explicit ExpVolume(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets, 
                       Plato::DataMap aDataMap) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Experimental Volume"),
            mProjectionFunction(),
            mPenaltyFunction(3.0, 0.0),
            mApplyProjection(mProjectionFunction)
    /**************************************************************************/
    {
         Plato::DegreeOneCubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
         mCubatureRule = tCubatureFactory.create(aMesh);
    }

    /**************************************************************************/
    ~ExpVolume()
    {
    }
    /**************************************************************************/

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> &,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
                  /**************************************************************************/
    {
        Plato::ComputeCellVolume<EvaluationType::SpatialDim> tComputeCellVolume;

        auto tNumCells = aControl.extent(0);
        auto & tApplyProjection = mApplyProjection;
        auto & tPenaltyFunction = mPenaltyFunction;
        auto tQuadratureWeight = mCubatureRule->getCubWeights();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            ConfigScalarType tCellVolume;
            tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
            tCellVolume *= tQuadratureWeight(aCellOrdinal);
            aResult(aCellOrdinal) = tCellVolume;

            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            ControlScalarType tPenaltyValue = tPenaltyFunction(tCellDensity);
            aResult(aCellOrdinal) *= tPenaltyValue;
        },"Experimental Volume");
    }
};
// class ExpVolume

} // namespace Plato

#endif /* PLATO_EXPVOLUME_HPP_ */
