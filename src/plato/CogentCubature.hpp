/*
 * CogentCubature.hpp
 *
 *  Created on: May 2, 2019
 */

#ifndef COGENT_CUBATURE_HPP_
#define COGENT_CUBATURE_HPP_

#include <string>

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/CubatureRule.hpp"
#include "plato/ImplicitFunctors.hpp"

#include <Cogent_IntegratorFactory.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>


namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CogentCubature : public Plato::CubatureRule<SpaceDim>
/******************************************************************************/
{
public:
    /******************************************************************************/
    CogentCubature(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams
    ) : 
      Plato::CubatureRule<SpaceDim>(aMesh),
      mNumCogPtsPerCell(0),
      mEnforceMinWeight(false),
      mMinWeight(0.0)
    /******************************************************************************/
    {
        this->initialize(aMesh, aInputParams);
    }
    /******************************************************************************/
    ~CogentCubature()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    void
    getNodeCoords( Plato::ScalarArray3D aNodes, Omega_h::Mesh& aMesh )
    /******************************************************************************/
    {
        auto tNumElems = aNodes.extent(0);
        auto tNumVertsPerCell = aNodes.extent(1);
        Plato::NodeCoordinate<SpaceDim> nodeCoords(&aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), LAMBDA_EXPRESSION(Plato::OrdinalType elemIndex)
        {
            for(int nodeIndex=0; nodeIndex < tNumVertsPerCell; nodeIndex++)
            {
                for(int dimIndex=0; dimIndex < SpaceDim; dimIndex++)
                {
                    aNodes(elemIndex, nodeIndex, dimIndex) = nodeCoords(elemIndex, nodeIndex, dimIndex);
                }
            }
        }, "node coordinates");
    }
        

private:
    /****************************************************************************//**/
    /*!
     * 
     ********************************************************************************/
    void initialize(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams
    )
    {
        Cogent::IntegratorFactory tCogentFactory;
        Teuchos::RCP<Intrepid2::Basis<Kokkos::Serial, Plato::Scalar>> tIntrepidBasis;
        Teuchos::RCP<shards::CellTopology> tBlockTopology;

        if(SpaceDim == 3)
        {
            tIntrepidBasis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<Kokkos::Serial, Plato::Scalar, Plato::Scalar>() );
            tBlockTopology = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<4> >() ) );
        } else 
        if(SpaceDim == 2)
        {
            tIntrepidBasis = Teuchos::rcp(new Intrepid2::Basis_HGRAD_TRI_C1_FEM<Kokkos::Serial, Plato::Scalar, Plato::Scalar>() );
            tBlockTopology = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );
        } else 
        if(SpaceDim == 1)
        {
            std::ostringstream tMsg;
            tMsg << " Fatal Error -- Requested Cogent cubature rule for 1D problem." << std::endl;
            throw std::invalid_argument(tMsg.str().c_str());
        }

        // parse and create Cogent cubature
        Teuchos::ParameterList tGeomSpec("Geometry");
        auto xmlFileName = aInputParams.get<std::string>("Model");
        Teuchos::updateParametersFromXmlFile(xmlFileName, Teuchos::ptrFromRef(tGeomSpec));

        bool mEnforceMinWeight = aInputParams.isType<double>("Minimum Weight");
        if( mEnforceMinWeight )
        {
            mMinWeight = aInputParams.get<double>("Minimum Weight");
        }

        mCogentCubature = tCogentFactory.create(tBlockTopology, tIntrepidBasis, tGeomSpec);

        // get standard point locations
        Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial> tPoints("points", 0, 0);
        mCogentCubature->getStandardPoints(tPoints);

        int tNumElems = aMesh.nelems();
        int tNumDims = tPoints.dimension(1);
        int tNumVertsPerCell = this->mBasisFunctions.extent(0);
        mNumCogPtsPerCell = tPoints.dimension(0);

        if(SpaceDim != tNumDims)
        {
            std::ostringstream tMsg;
            tMsg << " Fatal Error -- Dimensions don't match. Simulation: " 
                 << SpaceDim << ", Cubature: " << tNumDims << std::endl;
            throw std::invalid_argument(tMsg.str().c_str());
        }
        
        // Cogent is host-only.  Bring the mesh node positions to the host
        Plato::ScalarArray3D tNodesDevice("nodes", tNumElems, tNumVertsPerCell, SpaceDim);
   
        getNodeCoords(tNodesDevice, aMesh);
  
        auto tNodesHost = Kokkos::create_mirror(tNodesDevice);
        
        Kokkos::deep_copy(tNodesHost, tNodesDevice);

        // compute the gauss weights
        auto tCoordVals = Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>("coords",  tNumVertsPerCell, SpaceDim);
        auto tWeights   = Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>("weights", mNumCogPtsPerCell);
        auto tCubWeights = Kokkos::create_mirror(this->mCubWeights);
        for( int elemIndex=0; elemIndex<tNumElems; elemIndex++)
        {
            for( int nodeIndex=0; nodeIndex<tNumVertsPerCell; nodeIndex++)
            {
                for( int dimIndex=0; dimIndex<SpaceDim; dimIndex++)
                {
                    tCoordVals(nodeIndex, dimIndex) = tNodesHost(elemIndex, nodeIndex, dimIndex);
                }
            }

            mCogentCubature->getCubatureWeights(tWeights, tCoordVals);

            tCubWeights(elemIndex) = 0.0;
            for( int ptIndex=0; ptIndex<mNumCogPtsPerCell; ptIndex++)
            {
                tCubWeights(elemIndex) += tWeights(ptIndex);
            }
        }
        if( mEnforceMinWeight )
        {
            for( int elemIndex=0; elemIndex<tNumElems; elemIndex++)
            {
                if( tCubWeights(elemIndex) < mMinWeight )
                {
                    tCubWeights(elemIndex) = mMinWeight;
                }
            }
        }
        Kokkos::deep_copy(this->mCubWeights, tCubWeights);
    }

    Teuchos::RCP<Cogent::Integrator> mCogentCubature;
    int mNumCogPtsPerCell;
    bool mEnforceMinWeight;
    double mMinWeight;
};
// class CogentCubature

} // namespace Plato

#ifdef PLATO_1D
extern template class Plato::CogentCubature<1>;
#endif

#ifdef PLATO_2D
extern template class Plato::CogentCubature<2>;
#endif

#ifdef PLATO_3D
extern template class Plato::CogentCubature<3>;
#endif

#endif /* COGENT_CUBATURE_HPP_ */
