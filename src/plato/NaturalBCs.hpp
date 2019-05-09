#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

#include "ImplicitFunctors.hpp"

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/CogentCubature.hpp"

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Class for conformal natural boundary conditions.
  */
    template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
    class NaturalBCBase
  /******************************************************************************/
  {
      protected:
          const std::string        mName;
          const std::string        mSidesetName;
          Omega_h::Vector<NumDofs> mFlux;

    public:

    decltype(mSidesetName) const& get_ss_name() const { return mSidesetName; }

    decltype(mFlux) get_value() const { return mFlux; }


    NaturalBCBase(const std::string& name, Teuchos::ParameterList &params) :
    mName(name),
    mSidesetName(params.get<std::string>("Sides"))
    {
        const std::string type = params.get<std::string>("Type");
        if ("Uniform" == type) 
        {
            bool tValues = params.isType<Teuchos::Array<double>>("Values");
            bool tValue  = params.isType<double>("Value");
            if ( tValues && tValue )
            {
                TEUCHOS_TEST_FOR_EXCEPTION(true, 
                   std::logic_error,
                   " Natural Boundary Condition: provide EITHER 'Values' OR 'Value' Parameter.");
            } else 
            if ( tValues )
            {
                auto values = params.get<Teuchos::Array<double>>("Values");
                for(int i=0; i<NumDofs; i++) mFlux(i) = values[i];
            } else 
            if ( tValue )
            {
                Teuchos::Array<double> fluxVector(NumDofs, 0.0);
                mFlux(0) = params.get<double>("Value");
            } else {
                TEUCHOS_TEST_FOR_EXCEPTION(true, 
                   std::logic_error,
                   " Natural Boundary Condition: provide either 'Values' or 'Value' Parameter.");
            }
        }
        else 
        if ("Uniform Component" == type)
        {
            Teuchos::Array<double> values(NumDofs, 0.0);
            auto fluxComponent = params.get<std::string>("Component");
            auto value = params.get<double>("Value");
            if( (fluxComponent == "x" || fluxComponent == "X") ) values[0] = value;
            else
            if( (fluxComponent == "y" || fluxComponent == "Y") && DofsPerNode > 1 ) values[1] = value;
            else
            if( (fluxComponent == "z" || fluxComponent == "Z") && DofsPerNode > 2 ) values[2] = value;
            for(int i=0; i<NumDofs; i++) mFlux(i) = values[i];
        } else {
            TEUCHOS_TEST_FOR_EXCEPTION(true, 
               std::logic_error,
               " Natural Boundary Condition type invalid");
        }
    }
  };

  /******************************************************************************/
  /*!
    \brief Class for conformal natural boundary conditions.
  */
    template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
    class ConformalNaturalBC : public NaturalBCBase<SpatialDim, NumDofs, DofsPerNode, DofOffset>
  /******************************************************************************/
  {

  public:
  
    ConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string &aName, Teuchos::ParameterList &aParams) :
    NaturalBCBase<SpatialDim, NumDofs, DofsPerNode, DofOffset>(aName, aParams) {}

    ~ConformalNaturalBC(){}

    /*!
      \brief Get the contribution to the assembled forcing vector.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.

      The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
      \f{eqnarray*}{
          \phi(\xi,\psi)=
            \left\{ 
             \begin{array}{ccc} 
               N_I\left(\xi,\psi\right) x_I &
               N_I\left(\xi,\psi\right) y_I &
               N_I\left(\xi,\psi\right) z_I 
             \end{array} 
            \right\} \\
          f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i 
                \left|\left| 
                  \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi} 
                \right|\right| d\xi d\psi
      \f}
    */

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType> result,
              Plato::Scalar scale) const;

  };
  /******************************************************************************/
  /*!
    \brief Class for non-conformal natural boundary conditions.
  */
    template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
    class NonConformalNaturalBC : public NaturalBCBase<SpatialDim, NumDofs, DofsPerNode, DofOffset>
  /******************************************************************************/
  {

      std::shared_ptr<Plato::CubatureRule<SpatialDim>> mCubature;
    
  public:
  
    NonConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(
      const std::string &aName, 
      Teuchos::ParameterList &aParams, 
      Omega_h::Mesh& aMesh
    ) :
    NaturalBCBase<SpatialDim, NumDofs, DofsPerNode, DofOffset>(aName, aParams)
    {
        // create cogent integrator
        auto& cogentList = aParams.sublist("Cogent");
        mCubature = std::make_shared<Plato::CogentCubature<SpatialDim>>(aMesh, cogentList);
    }
  
    ~NonConformalNaturalBC(){}

    template<typename StateScalarType,
             typename ControlScalarType,
             typename ResultScalarType>
    void get( Omega_h::Mesh* aMesh,             
              const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType> result,
              Plato::Scalar scale) const;

  };
  
  
  /******************************************************************************/
  /*!
    \brief Owner class that contains a vector of NaturalBC objects.
  */
  template<int SpatialDim, int NumDofs=SpatialDim, int DofsPerNode=NumDofs, int DofOffset=0>
  class NaturalBCs
  /******************************************************************************/
  {
  private:
    Omega_h::Mesh& mMesh;

    std::vector<std::shared_ptr<ConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> m_cBCs;

    std::vector<std::shared_ptr<NonConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> m_ncBCs;

  public :

    /*!
      \brief Constructor that parses and creates a vector of NaturalBC objects
      based on the ParameterList.
    */
    NaturalBCs(Omega_h::Mesh &aMesh, Teuchos::ParameterList &params);

    /*!
      \brief Get the contribution to the assembled forcing vector from the owned boundary conditions.
      @param aMesh Omega_h mesh that contains sidesets.
      @param aMeshSets Omega_h mesh sets that contains sideset information.
      @param aResult Assembled vector to which the boundary terms will be added.
    */
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ResultScalarType>
    void get( const Omega_h::MeshSets& aMeshSets,
              Plato::ScalarMultiVectorT<  StateScalarType>,
              Plato::ScalarMultiVectorT<ControlScalarType>,
              Plato::ScalarMultiVectorT< ResultScalarType> result,
              Plato::Scalar scale = 1.0) const;

  };

  /**************************************************************************/
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ResultScalarType>
  void ConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get( Omega_h::Mesh* aMesh,             
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarMultiVectorT<  StateScalarType>,
                                               Plato::ScalarMultiVectorT<ControlScalarType>,
                                               Plato::ScalarMultiVectorT< ResultScalarType> result,
                                               Plato::Scalar scale) const
  /**************************************************************************/
  {
    // get sideset faces
    auto& sidesets = aMeshSets[Omega_h::SIDE_SET];
    auto ssIter = sidesets.find(this->mSidesetName);
    auto faceLids = (ssIter->second);
    auto numFaces = faceLids.size();


    // get mesh vertices
    auto face2verts = aMesh->ask_verts_of(SpatialDim-1);
    auto cell2verts = aMesh->ask_elem_verts();

    auto face2elems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto face2elems_map   = face2elems.a2ab;
    auto face2elems_elems = face2elems.ab2b;

    auto nodesPerFace = SpatialDim;
    auto nodesPerCell = SpatialDim+1;

    // create functor for accessing side node coordinates
    Plato::SideNodeCoordinate<SpatialDim> sideNodeCoordinate(aMesh);
    
    auto flux = this->mFlux;
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numFaces), LAMBDA_EXPRESSION(int iFace)
    {

      auto faceOrdinal = faceLids[iFace];

      // integrate
      //
      Omega_h::Matrix<SpatialDim, SpatialDim-1> jacobian;
      for (int d1=0; d1<SpatialDim-1; d1++)
      {
        for (int d2=0; d2<SpatialDim; d2++)
        { 
          jacobian[d1][d2] = sideNodeCoordinate(faceOrdinal,d1,d2) - sideNodeCoordinate(faceOrdinal,SpatialDim-1,d2);
        }
      }

      Scalar weight(0.0);
      if(SpatialDim==1){
        weight=1.0;
      } else
      if(SpatialDim==2){
        weight = 1.0/2.0*sqrt(jacobian[0][0]*jacobian[0][0]+jacobian[0][1]*jacobian[0][1]);
      } else
      if(SpatialDim==3){
        auto a1 = jacobian[0][1]*jacobian[1][2]-jacobian[0][2]*jacobian[1][1];
        auto a2 = jacobian[0][2]*jacobian[1][0]-jacobian[0][0]*jacobian[1][2];
        auto a3 = jacobian[0][0]*jacobian[1][1]-jacobian[0][1]*jacobian[1][0];
        weight = 1.0/6.0*sqrt(a1*a1+a2*a2+a3*a3);
      }

      int localNodeOrd[SpatialDim];
      for( int localElemOrd = face2elems_map[faceOrdinal]; 
               localElemOrd < face2elems_map[faceOrdinal+1]; ++localElemOrd ){
        auto cellOrdinal = face2elems_elems[localElemOrd];
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int jNode=0; jNode<nodesPerCell; jNode++){
            if( face2verts[faceOrdinal*nodesPerFace+iNode] == cell2verts[cellOrdinal*nodesPerCell + jNode] ) localNodeOrd[iNode] = jNode;
          }
        }
        for( int iNode=0; iNode<nodesPerFace; iNode++){
          for( int iDof=0; iDof<NumDofs; iDof++){
            auto cellDofOrdinal = localNodeOrd[iNode] * DofsPerNode + iDof + DofOffset;
            result(cellOrdinal,cellDofOrdinal) += weight*flux[iDof];
          }
        }
      }
    });
  }

  /**************************************************************************/
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ResultScalarType>
  void NonConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get( Omega_h::Mesh* aMesh,             
                                               const Omega_h::MeshSets& aMeshSets,
                                               Plato::ScalarMultiVectorT<  StateScalarType>,
                                               Plato::ScalarMultiVectorT<ControlScalarType>,
                                               Plato::ScalarMultiVectorT< ResultScalarType> result,
                                               Plato::Scalar scale) const
  /**************************************************************************/
  {

      auto nodesPerCell = SpatialDim+1;
      auto numElems = aMesh->nelems();

      auto flux = this->mFlux;
      auto weights = mCubature->getCubWeights();
      auto basis = mCubature->getBasisFunctions();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numElems), LAMBDA_EXPRESSION(int iElem)
      {
          for( int iNode=0; iNode<nodesPerCell; iNode++)
          {
              for( int iDof=0; iDof<NumDofs; iDof++)
              {
                  auto cellDofOrdinal = iNode * DofsPerNode + iDof + DofOffset;
                  result(iElem, cellDofOrdinal) += basis(iNode) * weights(iElem) * flux[iDof];
              }
          }
      });
  }


  /****************************************************************************/
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::
  NaturalBCs(Omega_h::Mesh &aMesh, Teuchos::ParameterList &params) :
    mMesh(aMesh),
    m_cBCs(),
    m_ncBCs()
  /****************************************************************************/
  {
    for (Teuchos::ParameterList::ConstIterator i = params.begin(); i != params.end(); ++i) {
      const Teuchos::ParameterEntry &entry = params.entry(i);
      const std::string             &name  = params.name(i);
  
      TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(),
         std::logic_error,
         "Parameter in Boundary Conditions block not valid.  Expect lists only.");

      Teuchos::ParameterList& sublist = params.sublist(name);

      std::string geometryType("Mesh");
      if( sublist.isType<std::string>("Geometry") )
      {
          geometryType = sublist.get<std::string>("Geometry");
      }

      if( geometryType == "Mesh" )
      {
          m_cBCs.push_back(
            std::make_shared<ConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>(name, sublist));
      } else 
      if( geometryType == "Cogent" )
      {
          m_ncBCs.push_back(
            std::make_shared<NonConformalNaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>(name, sublist, mMesh));
      }
    }
  }

  /**************************************************************************/
  /*!
    \brief Add the boundary load to the result workset
  */
  template<int SpatialDim, int NumDofs, int DofsPerNode, int DofOffset>
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ResultScalarType>
  void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get(
       const Omega_h::MeshSets& aMeshSets, 
       Kokkos::View<   StateScalarType**, Kokkos::LayoutRight, Plato::MemSpace > state,
       Kokkos::View< ControlScalarType**, Kokkos::LayoutRight, Plato::MemSpace > control,
       Kokkos::View<  ResultScalarType**, Kokkos::LayoutRight, Plato::MemSpace > result,
       Plato::Scalar scale) const
  /**************************************************************************/
  {
    for (const auto &bc : m_cBCs){
      bc->get(&mMesh, aMeshSets, state, control, result, scale);
    }

    for (const auto &bc : m_ncBCs){
      bc->get(&mMesh, aMeshSets, state, control, result, scale);
    }
  }


} // end namespace Plato 

#endif

