#ifndef HEAT_EQUATION_RESIDUAL_HPP
#define HEAT_EQUATION_RESIDUAL_HPP

#include "plato/SimplexThermal.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/ScalarGrad.hpp"
#include "plato/ThermalFlux.hpp"
#include "plato/ThermalContent.hpp"
#include "plato/FluxDivergence.hpp"
#include "plato/SimplexFadTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/PlatoCubatureFactory.hpp"
#include "plato/Simp.hpp"
#include "plato/Ramp.hpp"
#include "plato/Heaviside.hpp"

#include "plato/LinearThermalMaterial.hpp"
#include "plato/AbstractVectorFunctionInc.hpp"
#include "plato/ImplicitFunctors.hpp"
#include "plato/InterpolateFromNodal.hpp"
#include "plato/ProjectToNode.hpp"
#include "plato/ApplyWeighting.hpp"
#include "plato/NaturalBCs.hpp"
#include "plato/SimplexFadTypes.hpp"

#include "plato/ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class HeatEquationResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::AbstractVectorFunctionInc<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::m_numDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::m_numDofsPerNode;

    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMesh;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::m_dataMap;
    using Plato::AbstractVectorFunctionInc<EvaluationType>::mMeshSets;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;



    Omega_h::Matrix< SpaceDim, SpaceDim> m_cellConductivity;
    Plato::Scalar m_cellDensity;
    Plato::Scalar m_cellSpecificHeat;
    
    IndicatorFunctionType m_indicatorFunction;
    Plato::ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> m_applyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim,m_numDofsPerNode,IndicatorFunctionType> m_applyMassWeighting;

    std::shared_ptr<Plato::CubatureRule<EvaluationType::SpatialDim>> m_cubatureRule;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>> m_boundaryLoads;

  public:
    /**************************************************************************/
    HeatEquationResidual(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Plato::DataMap& aDataMap,
      Teuchos::ParameterList& aProblemParams,
      Teuchos::ParameterList& aPenaltyParams) :
     AbstractVectorFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, {"Temperature"}),
     m_indicatorFunction(aPenaltyParams),
     m_applyFluxWeighting(m_indicatorFunction),
     m_applyMassWeighting(m_indicatorFunction),
     m_boundaryLoads(nullptr)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<SpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      m_cellConductivity = materialModel->getConductivityMatrix();
      m_cellDensity      = materialModel->getMassDensity();
      m_cellSpecificHeat = materialModel->getSpecificHeat();


      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Natural Boundary Conditions"))
      {
          m_boundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,m_numDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
      }

       Plato::CubatureFactory<EvaluationType::SpatialDim>  tCubatureFactory;
       m_cubatureRule = tCubatureFactory.create(aMesh, aProblemParams);
    
    }


    /**************************************************************************/
    void
    evaluate( const Plato::ScalarMultiVectorT< StateScalarType     > & aState,
              const Plato::ScalarMultiVectorT< PrevStateScalarType > & aPrevState,
              const Plato::ScalarMultiVectorT< ControlScalarType   > & aControl,
              const Plato::ScalarArray3DT    < ConfigScalarType    > & aConfig,
                    Plato::ScalarMultiVectorT< ResultScalarType    > & aResult,
                    Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using PrevGradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, PrevStateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Plato::ScalarMultiVectorT<GradScalarType> tGrad("temperature gradient at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<PrevGradScalarType> tPrevGrad("temperature gradient at step k-1",numCells,SpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient",numCells,m_numNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> tFlux("thermal flux at step k",numCells,SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tPrevFlux("thermal flux at step k-1",numCells,SpaceDim);

      Plato::ScalarVectorT<StateScalarType> tTemperature("Gauss point temperature at step k", numCells);
      Plato::ScalarVectorT<PrevStateScalarType> tPrevTemperature("Gauss point temperature at step k-1", numCells);

      Plato::ScalarVectorT<ResultScalarType> tThermalContent("Gauss point heat content at step k", numCells);
      Plato::ScalarVectorT<ResultScalarType> tPrevThermalContent("Gauss point heat content at step k-1", numCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  computeGradient;

      Plato::ScalarGrad<SpaceDim>            scalarGrad;
      Plato::ThermalFlux<SpaceDim>           thermalFlux(m_cellConductivity);
      Plato::FluxDivergence<SpaceDim>        fluxDivergence;

      Plato::InterpolateFromNodal<SpaceDim, m_numDofsPerNode> interpolateFromNodal;
      Plato::ThermalContent thermalContent(m_cellDensity, m_cellSpecificHeat);
      Plato::ProjectToNode<SpaceDim, m_numDofsPerNode> projectThermalContent;
      
      auto basisFunctions = m_cubatureRule->getBasisFunctions();
    
      auto& applyFluxWeighting  = m_applyFluxWeighting;
      auto& applyMassWeighting  = m_applyMassWeighting;
      auto quadratureWeight = m_cubatureRule->getCubWeights();
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
    
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight(cellOrdinal);
    
        // compute temperature gradient
        //
        scalarGrad(cellOrdinal, tGrad, aState, gradient);
        scalarGrad(cellOrdinal, tPrevGrad, aPrevState, gradient);
    
        // compute flux
        //
        thermalFlux(cellOrdinal, tFlux, tGrad);
        thermalFlux(cellOrdinal, tPrevFlux, tPrevGrad);
    
        // apply weighting
        //
        applyFluxWeighting(cellOrdinal, tFlux, aControl);
        applyFluxWeighting(cellOrdinal, tPrevFlux, aControl);

        // compute stress divergence
        //
        fluxDivergence(cellOrdinal, aResult, tFlux,     gradient, cellVolume, aTimeStep/2.0);
        fluxDivergence(cellOrdinal, aResult, tPrevFlux, gradient, cellVolume, aTimeStep/2.0);


        // add capacitance terms
        
        // compute temperature at gausspoints
        //
        interpolateFromNodal(cellOrdinal, basisFunctions, aState, tTemperature);
        interpolateFromNodal(cellOrdinal, basisFunctions, aPrevState, tPrevTemperature);

        // compute the specific heat content (i.e., specific heat times temperature)
        //
        thermalContent(cellOrdinal, tThermalContent, tTemperature);
        thermalContent(cellOrdinal, tPrevThermalContent, tPrevTemperature);

        // apply weighting
        //
        applyMassWeighting(cellOrdinal, tThermalContent, aControl);
        applyMassWeighting(cellOrdinal, tPrevThermalContent, aControl);

        // project to nodes
        //
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, aResult);
        projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tPrevThermalContent, aResult, -1.0);

      },"flux divergence");

      if( m_boundaryLoads != nullptr )
      {
          m_boundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aResult, -aTimeStep/2.0 );
          m_boundaryLoads->get( &mMesh, mMeshSets, aPrevState, aControl, aResult, -aTimeStep/2.0 );
      }
    }
};
// class HeatEquationResidual

} // namespace Plato

#ifdef PLATO_1D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEC_INC(Plato::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif

#endif
