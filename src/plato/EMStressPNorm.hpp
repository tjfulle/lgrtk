#ifndef TM_STRESS_P_NORM_HPP
#define TM_STRESS_P_NORM_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/LinearElectroelasticMaterial.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/EMKinematics.hpp"
#include "plato/EMKinetics.hpp"
#include "plato/TensorPNorm.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/PlatoCubatureFactory.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class EMStressPNorm : 
  public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexElectromechanics<SpaceDim>::m_numVoigtTerms;
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerCell;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> m_materialModel;

    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<SpaceDim,m_numVoigtTerms,IndicatorFunctionType> m_applyWeighting;
    std::shared_ptr<Plato::CubatureRule<EvaluationType::SpatialDim>> m_CubatureRule;

    Teuchos::RCP<TensorNormBase<m_numVoigtTerms,EvaluationType>> m_norm;

  public:
    /**************************************************************************/
    EMStressPNorm(Omega_h::Mesh& aMesh,
                  Omega_h::MeshSets& aMeshSets,
                  Plato::DataMap& aDataMap, 
                  Teuchos::ParameterList& aProblemParams, 
                  Teuchos::ParameterList& aPenaltyParams) :
              Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Stress P-Norm"),
              m_indicatorFunction(aPenaltyParams),
              m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      Plato::ElectroelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      m_materialModel = mmfactory.create();

      auto params = aProblemParams.get<Teuchos::ParameterList>("Stress P-Norm");

      TensorNormFactory<m_numVoigtTerms, EvaluationType> normFactory;
      m_norm = normFactory.create(params);

      Plato::CubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
      m_CubatureRule = tCubatureFactory.create(aMesh, aProblemParams);
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType = 
        typename Plato::fad_type_t<Plato::SimplexElectromechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::EMKinematics<SpaceDim>                  kinematics;
      Plato::EMKinetics<SpaceDim>                    kinetics(m_materialModel);

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight", numCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", numCells, m_numNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", numCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> efield("efield", numCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", numCells, m_numVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> edisp ("edisp",  numCells, SpaceDim);

      auto quadratureWeight = m_CubatureRule->getCubWeights();
      auto applyWeighting   = m_applyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight(cellOrdinal);

        // compute strain
        //
        kinematics(cellOrdinal, strain, efield, aState, gradient);

        // compute stress
        //
        kinetics(cellOrdinal, stress, edisp, strain, efield);
      
        // apply weighting
        //
        applyWeighting(cellOrdinal, stress, aControl);

      },"Compute Stress");

      m_norm->evaluate(aResult, stress, aControl, cellVolume);

    }

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      m_norm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      m_norm->postEvaluate(resultValue);
    }
};
// class EMStressPNorm

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 3)
#endif

#endif
