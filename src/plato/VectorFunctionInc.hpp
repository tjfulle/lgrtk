#ifndef VECTOR_FUNCTION_INC_HPP
#define VECTOR_FUNCTION_INC_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractVectorFunctionInc.hpp"
#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, U^{k-1}, X)

   and manages the evaluation of the function and derivatives wrt state, U^k; 
   previous state, U^{k-1}; and control, X.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunctionInc : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using JacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianP;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::shared_ptr<Plato::AbstractVectorFunctionInc<Residual>>  mVectorFunctionResidual;
    std::shared_ptr<Plato::AbstractVectorFunctionInc<Jacobian>>  mVectorFunctionJacobianU;
    std::shared_ptr<Plato::AbstractVectorFunctionInc<JacobianP>> mVectorFunctionJacobianP;
    std::shared_ptr<Plato::AbstractVectorFunctionInc<GradientX>> mVectorFunctionJacobianX;
    std::shared_ptr<Plato::AbstractVectorFunctionInc<GradientZ>> mVectorFunctionJacobianZ;

    Plato::DataMap& mDataMap;

  public:

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aMeshSets mesh sets data base
    * @param [in] aDataMap problem-specific data map 
    * @param [in] aParamList Teuchos parameter list with input data
    * @param [in] aProblemType problem type 
    *
    ******************************************************************************/
    VectorFunctionInc(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap& aDataMap,
                   Teuchos::ParameterList& aParamList,
                   std::string& aProblemType) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mVectorFunctionResidual = tFunctionFactory.template createVectorFunctionInc<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianU = tFunctionFactory.template createVectorFunctionInc<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianP = tFunctionFactory.template createVectorFunctionInc<JacobianP>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianZ = tFunctionFactory.template createVectorFunctionInc<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionJacobianX = tFunctionFactory.template createVectorFunctionInc<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    VectorFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mVectorFunctionResidual(),
            mVectorFunctionJacobianU(),
            mVectorFunctionJacobianP(),
            mVectorFunctionJacobianX(),
            mVectorFunctionJacobianZ(),
            mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    * @param [in] aJacobian Jacobian evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::AbstractVectorFunctionInc<Residual>>& aResidual,
                          const std::shared_ptr<Plato::AbstractVectorFunctionInc<Jacobian>>& aJacobian)
    {
        mVectorFunctionResidual = aResidual;
        mVectorFunctionJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<Plato::AbstractVectorFunctionInc<GradientZ>>& aGradientZ)
    {
        mVectorFunctionJacobianZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<Plato::AbstractVectorFunctionInc<GradientX>>& aGradientX)
    {
        mVectorFunctionJacobianX = aGradientX; 
    }

    /**************************************************************************//**
    *
    * @brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /**************************************************************************//**
    *
    * @brief Return state names
    *
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
      return mVectorFunctionResidual->getDofNames();
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aState,
          const Plato::ScalarVector & aPrevState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Residual::ConfigScalarType;
      using StateScalar     = typename Residual::StateScalarType;
      using PrevStateScalar = typename Residual::PrevStateScalarType;
      using ControlScalar   = typename Residual::ControlScalarType;
      using ResultScalar    = typename Residual::ResultScalarType;

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // create result
      //
      Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual",mNumCells, mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionResidual->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

      // create and assemble to return view
      //
      Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tReturnValue("Assembled Residual",mNumDofsPerNode*mNumNodes);
      Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue );

      return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aPrevState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using PrevStateScalar = typename GradientX::PrevStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset prev state
        //
        Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

        // evaluate function
        //
        mVectorFunctionJacobianX->evaluate(tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        // create return matrix
        //
        auto tMesh = mVectorFunctionJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aPrevState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename Jacobian::ConfigScalarType;
      using StateScalar     = typename Jacobian::StateScalarType;
      using PrevStateScalar = typename Jacobian::PrevStateScalarType;
      using ControlScalar   = typename Jacobian::ControlScalarType;
      using ResultScalar    = typename Jacobian::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      // 
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionJacobianU->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_p(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aPrevState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename JacobianP::ConfigScalarType;
      using StateScalar     = typename JacobianP::StateScalarType;
      using PrevStateScalar = typename JacobianP::PrevStateScalarType;
      using ControlScalar   = typename JacobianP::ControlScalarType;
      using ResultScalar    = typename JacobianP::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      // 
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      // 
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionJacobianP->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aState,
               const Plato::ScalarVectorT<Plato::Scalar> & aPrevState,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar    = typename GradientZ::ConfigScalarType;
      using StateScalar     = typename GradientZ::StateScalarType;
      using PrevStateScalar = typename GradientZ::PrevStateScalarType;
      using ControlScalar   = typename GradientZ::ControlScalarType;
      using ResultScalar    = typename GradientZ::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset prev state
      //
      Plato::ScalarMultiVectorT<PrevStateScalar> tPrevStateWS("Prev State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aPrevState, tPrevStateWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumDofsPerCell);

      // evaluate function 
      //
      mVectorFunctionJacobianZ->evaluate( tStateWS, tPrevStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return (tJacobianMat);
    }
};
// class VectorFunctionInc

} // namespace Plato

#endif
