#ifndef EFFECTIVE_ELASTIC_ENERGY_HPP
#define EFFECTIVE_ELASTIC_ENERGY_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/ScalarProduct.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/Strain.hpp"
#include "plato/LinearElasticMaterial.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/HomogenizedStress.hpp"
#include "plato/AbstractScalarFunction.hpp"
#include "plato/PlatoCubatureFactory.hpp"
#include "plato/ExpInstMacros.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal effective energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class EffectiveEnergy : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms;
    using Simplex<mSpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::m_numDofsPerCell;

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap;
    using Plato::AbstractScalarFunction<EvaluationType>::m_functionName;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,m_numVoigtTerms,IndicatorFunctionType> m_applyWeighting;
    std::shared_ptr<Plato::CubatureRule<EvaluationType::SpatialDim>> mCubatureRule;

    Omega_h::Matrix< m_numVoigtTerms, m_numVoigtTerms> m_cellStiffness;
    Omega_h::Vector<m_numVoigtTerms> m_assumedStrain;
    Plato::OrdinalType m_columnIndex;

  public:
    /**************************************************************************/
    EffectiveEnergy(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aProblemParams,
                    Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Effective Energy"),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellStiffness = materialModel->getStiffnessMatrix();

      Teuchos::ParameterList& tParams = aProblemParams.sublist(m_functionName);
      auto tAssumedStrain = tParams.get<Teuchos::Array<double>>("Assumed Strain");
      assert(tAssumedStrain.size() == m_numVoigtTerms);
      for( Plato::OrdinalType iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++)
      {
          m_assumedStrain[iVoigt] = tAssumedStrain[iVoigt];
      }

      // parse cell problem forcing
      //
      if(aProblemParams.isSublist("Cell Problem Forcing"))
      {
          m_columnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
      }
      else
      {
          // JR TODO: throw
      }
      Plato::CubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
      mCubatureRule = tCubatureFactory.create(aMesh, aProblemParams);
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

      Plato::Strain<mSpaceDim> voigtStrain;
      Plato::ScalarProduct<m_numVoigtTerms> scalarProduct;
      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::HomogenizedStress < mSpaceDim > homogenizedStress(m_cellStiffness, m_columnIndex);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        strain("strain",numCells,m_numVoigtTerms);

      Kokkos::View<ConfigScalarType***, Kokkos::LayoutRight, Plato::MemSpace>
        gradient("gradient",numCells,m_numNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        stress("stress",numCells,m_numVoigtTerms);

      auto quadratureWeight = mCubatureRule->getCubWeights();
      auto applyWeighting   = m_applyWeighting;
      auto assumedStrain    = m_assumedStrain;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight(aCellOrdinal);

        // compute strain
        //
        voigtStrain(aCellOrdinal, strain, aState, gradient);

        // compute stress
        //
        homogenizedStress(aCellOrdinal, stress, strain);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, stress, aControl);
    
        // compute element internal energy (inner product of strain and weighted stress)
        //
        scalarProduct(aCellOrdinal, aResult, stress, assumedStrain, cellVolume);

      },"energy gradient");
    }
};
// class EffectiveEnergy

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::EffectiveEnergy, Plato::SimplexMechanics, 3)
#endif

#endif
