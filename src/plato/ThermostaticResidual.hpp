#ifndef THERMOSTATIC_RESIDUAL_HPP
#define THERMOSTATIC_RESIDUAL_HPP

#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"
#include "plato/PlatoCubatureFactory.hpp"

#include "plato/LinearThermalMaterial.hpp"
#include "plato/AbstractVectorFunction.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermostaticResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::m_numDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::m_numDofsPerNode;

    using Plato::AbstractVectorFunction<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunction<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunction<EvaluationType>::mMeshSets;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< SpaceDim, SpaceDim> m_cellConductivity;
    
    IndicatorFunctionType m_indicatorFunction;
    ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> m_applyWeighting;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;

    std::shared_ptr<Plato::CubatureRule<EvaluationType::SpatialDim>> mCubatureRule;

  public:
    /**************************************************************************/
    ThermostaticResidual(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractVectorFunction<EvaluationType>(aMesh, aMeshSets, aDataMap),
            m_indicatorFunction(aPenaltyParams),
            m_applyWeighting(m_indicatorFunction),
            m_boundaryLoads(nullptr)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();

      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
      }
    
       Plato::CubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
       mCubatureRule = tCubatureFactory.create(aMesh, aProblemParams);

    }


    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT<StateScalarType  > & state,
              const Plato::ScalarMultiVectorT<ControlScalarType> & control,
              const Plato::ScalarArray3DT<ConfigScalarType > & config,
              Plato::ScalarMultiVectorT<ResultScalarType > & result,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      Kokkos::deep_copy(result, 0.0);

      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tgrad("temperature gradient",numCells,SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace>
        tflux("thermal flux",numCells,SpaceDim);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;

      Plato::ScalarGrad<SpaceDim>            scalarGrad;
      Plato::ThermalFlux<SpaceDim>           thermalFlux(m_cellConductivity);
      Plato::FluxDivergence<SpaceDim>        fluxDivergence;
    
      auto& applyWeighting  = m_applyWeighting;
      auto quadratureWeight = mCubatureRule->getCubWeights();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
    
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight(cellOrdinal);
    
        // compute temperature gradient
        //
        scalarGrad(cellOrdinal, tgrad, state, gradient);
    
        // compute flux
        //
        thermalFlux(cellOrdinal, tflux, tgrad);
    
        // apply weighting
        //
        applyWeighting(cellOrdinal, tflux, control);
    
        // compute stress divergence
        //
        fluxDivergence(cellOrdinal, result, tflux, gradient, cellVolume);
        
      },"flux divergence");

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, state, control, result );
      }
    }
};
// class ThermostaticResidual

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC(Plato::ThermostaticResidual, Plato::SimplexThermal, 3)
#endif

#endif
