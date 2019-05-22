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
class CogentCubatureBase
/******************************************************************************/
{
public:
    /******************************************************************************/
    CogentCubatureBase() :
      mNumCogPtsPerElem(0)
    {}
    /******************************************************************************/

    /******************************************************************************/
    CogentCubatureBase(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams
    ) :
      mNumCogPtsPerElem(0)
    /******************************************************************************/
    {
        this->initialize(aMesh, aInputParams);
    }
    /******************************************************************************/
    ~CogentCubatureBase()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    Plato::ScalarArray3D
    getNodeCoords( Omega_h::Mesh& aMesh )
    /******************************************************************************/
    {
        int tNumElems = aMesh.nelems();
        int tNumVertsPerElem = SpaceDim+1;

        Plato::ScalarArray3D tNodesDevice("nodes", tNumElems, tNumVertsPerElem, SpaceDim);

        Plato::NodeCoordinate<SpaceDim> nodeCoords(&aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), LAMBDA_EXPRESSION(Plato::OrdinalType elemIndex)
        {
            for(int nodeIndex=0; nodeIndex < tNumVertsPerElem; nodeIndex++)
            {
                for(int dimIndex=0; dimIndex < SpaceDim; dimIndex++)
                {
                    tNodesDevice(elemIndex, nodeIndex, dimIndex) = nodeCoords(elemIndex, nodeIndex, dimIndex);
                }
            }
        }, "node coordinates");

        return tNodesDevice;
    }
        
protected:
    /****************************************************************************//**/
    /*!
     * 
     ********************************************************************************/
    void initialize(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams
    ){
        auto aModelName = aInputParams.get<std::string>("Model");
        createCogentCubature(aModelName);
        computeCubatureWeights(aMesh);
    }

private:
    /****************************************************************************/
    void computeCubatureWeights( Omega_h::Mesh& aMesh )
    /****************************************************************************/
    {
        auto tNodesDevice = getNodeCoords(aMesh);
        auto tNodesHost = Kokkos::create_mirror(tNodesDevice);
        Kokkos::deep_copy(tNodesHost, tNodesDevice);

        // compute the gauss weights
        int tNumElems = aMesh.nelems();
        int tNumVertsPerElem = SpaceDim+1;
        auto tCoordVals = Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>("coords",  tNumVertsPerElem, SpaceDim);
        auto tWeights   = Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>("weights", mNumCogPtsPerElem);
        mCogentWeights =
          Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>("Cogent Weights", tNumElems, mNumCogPtsPerElem);

        for( int elemIndex=0; elemIndex<tNumElems; elemIndex++)
        {
            for( int nodeIndex=0; nodeIndex<tNumVertsPerElem; nodeIndex++)
            {
                for( int dimIndex=0; dimIndex<SpaceDim; dimIndex++)
                {
                    tCoordVals(nodeIndex, dimIndex) = tNodesHost(elemIndex, nodeIndex, dimIndex);
                }
            }

            mCogentCubature->getCubatureWeights(tWeights, tCoordVals);

            for( int ptIndex=0; ptIndex<mNumCogPtsPerElem; ptIndex++)
            {
                mCogentWeights(elemIndex, ptIndex) = tWeights(ptIndex);
            }
        }
    }

    /****************************************************************************/
    void createCogentCubature(const std::string& aXmlFileName)
    /****************************************************************************/
    {
        Teuchos::ParameterList tGeomSpec("Geometry");
        Teuchos::updateParametersFromXmlFile(aXmlFileName, Teuchos::ptrFromRef(tGeomSpec));

        auto tIntrepidBasis = createBasis();
        auto tBlockTopology = createTopology();

        Cogent::IntegratorFactory tCogentFactory;
        mCogentCubature = tCogentFactory.create(tBlockTopology, tIntrepidBasis, tGeomSpec);

        validateCogentCubature();

    }

    /****************************************************************************/
    Teuchos::RCP<Intrepid2::Basis<Kokkos::Serial, Plato::Scalar>>
    createBasis()
    /****************************************************************************/
    {
        if(SpaceDim == 3)
        {
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_TET_C1_FEM<Kokkos::Serial, Plato::Scalar, Plato::Scalar>() );
        } else 
        if(SpaceDim == 2)
        {
            return Teuchos::rcp(new Intrepid2::Basis_HGRAD_TRI_C1_FEM<Kokkos::Serial, Plato::Scalar, Plato::Scalar>() );
        } else 
        {
            throw std::invalid_argument("Cogent cubature only available for 2D and 3D");
        }
    }

    /****************************************************************************/
    Teuchos::RCP<shards::CellTopology>
    createTopology()
    /****************************************************************************/
    {
        if(SpaceDim == 3)
        {
            return Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Tetrahedron<4> >() ) );
        } else 
        if(SpaceDim == 2)
        {
            return Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );
        } else
        {
            throw std::invalid_argument("Cogent cubature only available for 2D and 3D");
        }
    }

    /****************************************************************************/
    void
    validateCogentCubature()
    /****************************************************************************/
    {
        // get standard point locations
        Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial> tPoints("points", 0, 0);
        mCogentCubature->getStandardPoints(tPoints);

        int tNumDims = tPoints.dimension(1);
        mNumCogPtsPerElem = tPoints.dimension(0);

        if(SpaceDim != tNumDims)
        {
            std::ostringstream tMsg;
            tMsg << " Fatal Error -- Dimensions don't match. Simulation: " 
                 << SpaceDim << ", Cubature: " << tNumDims << std::endl;
            throw std::invalid_argument(tMsg.str().c_str());
        }
    }

  protected:
    Teuchos::RCP<Cogent::Integrator> mCogentCubature;
    Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial> mCogentWeights;
    Plato::OrdinalType mNumCogPtsPerElem;
};




/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CogentCubatureDegreeOne : 
        public Plato::CubatureRuleDegreeOne<SpaceDim>,
        public Plato::CogentCubatureBase<SpaceDim>
/******************************************************************************/
{
public:
    /******************************************************************************/
    CogentCubatureDegreeOne(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams,
      const Plato::DataMap& dm = nullptr
    ) : 
      Plato::CubatureRuleDegreeOne<SpaceDim>(aMesh),
      mEnforceMinWeight(false),
      mMinWeight(0.0)
    /******************************************************************************/
    {
        auto aModelName = aInputParams.get<std::string>("Model");
        if( dm != nullptr && dm->scalarVectors.count(aModelName) )
        {
            this->mCubWeights = dm->scalarVectors[aModelName];
        }
        else
        {
            Plato::CogentCubatureBase<SpaceDim>::initialize(aMesh, aInputParams);
            this->initialize(aMesh, aInputParams);
            if( dm != nullptr )
            {
                dm->scalarVectors[aModelName] = this->mCubWeights;
            }
        }
    }
    /******************************************************************************/
    ~CogentCubatureDegreeOne()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    void
    applyScaling(
        Omega_h::Mesh& aMesh,
        Plato::ScalarVector aWeights
    )
    /******************************************************************************/
    {
        int tNumElems = aMesh.nelems();

        Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;
        Plato::NodeCoordinate<SpaceDim> tNodeCoordinate(&aMesh);
        Plato::ScalarArray3D tConfigWS("config workset", tNumElems, SpaceDim+1, SpaceDim);

        auto tCubWeights = this->mCubWeights;
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), LAMBDA_EXPRESSION(Plato::OrdinalType tElemIndex)
        {
            for(int tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
            {
                for(int tNodeIndex = 0; tNodeIndex < SpaceDim+1; tNodeIndex++)
                {
                    tConfigWS(tElemIndex, tNodeIndex, tDimIndex) = tNodeCoordinate(tElemIndex, tNodeIndex, tDimIndex);
                }
            }
            Plato::Scalar tCellVolume;
            tComputeCellVolume(tElemIndex, tConfigWS, tCellVolume);
            tCubWeights(tElemIndex) /= tCellVolume;
        }, "apply scaling");
    }

    /******************************************************************************/
    void applyLimits(
        Plato::ScalarVector aWeights
    )
    /******************************************************************************/
    {
        if( mEnforceMinWeight )
        {
            auto tMinWeight = mMinWeight;
            auto tNumElems = aWeights.extent(0);
            Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), LAMBDA_EXPRESSION(Plato::OrdinalType elemIndex)
            {
                if( aWeights(elemIndex) < tMinWeight )
                {
                    aWeights(elemIndex) = tMinWeight;
                }
            }, "apply limits");
        }
    }


private:
    /********************************************************************************/
    void initialize(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams)
    /********************************************************************************/
    {
        parseInput(aInputParams);

        auto tToWeights     = this->mCubWeights;
        auto tFromWeights   = this->mCogentWeights;

        sumCogentWeights(tToWeights, tFromWeights);

        applyLimits(tToWeights);
      
        applyScaling(aMesh, tToWeights);
    }

    /******************************************************************************/
    void sumCogentWeights(
        const Plato::ScalarVector& aToWeights,
        const Kokkos::DynRankView<Plato::Scalar, Kokkos::Serial>& aFromWeights
    )
    /******************************************************************************/
    {
        auto tToWeightsHost = Kokkos::create_mirror(aToWeights);
        auto tNumElems = aFromWeights.extent(0);
        auto tNumCogPtsPerElem = aFromWeights.extent(1);
        for( int elemIndex=0; elemIndex<tNumElems; elemIndex++)
        {
            tToWeightsHost(elemIndex) = 0.0;
            for( int ptIndex=0; ptIndex<tNumCogPtsPerElem; ptIndex++)
            {
                tToWeightsHost(elemIndex) += aFromWeights(elemIndex, ptIndex);
            }
        }
        Kokkos::deep_copy(aToWeights, tToWeightsHost);
    }

    /****************************************************************************/
    void parseInput(const Teuchos::ParameterList& aInputParams)
    /****************************************************************************/
    {
        mEnforceMinWeight = aInputParams.isType<double>("Minimum Weight");
        if( mEnforceMinWeight )
        {
            mMinWeight = aInputParams.get<double>("Minimum Weight");
        }
    }

  protected:
    bool mEnforceMinWeight;
    double mMinWeight;

};
// class CogentCubatureDegreeOne


/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class CogentCubature : 
        public Plato::CubatureRule<SpaceDim>,
        public Plato::CogentCubatureBase<SpaceDim>
/******************************************************************************/
{
public:
    /******************************************************************************/
    CogentCubature(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams,
      const Plato::DataMap& dm = nullptr
    ) : 
      Plato::CubatureRule<SpaceDim>(aMesh, aInputParams.get<int>("Order",2))
    /******************************************************************************/
    {
        auto aModelName = aInputParams.get<std::string>("Model");
        if( dm != nullptr && dm->scalarMultiVectors.count(aModelName) )
        {
            this->mCubWeights = dm->scalarMultiVectors[aModelName];
        }
        else
        {
            Plato::CogentCubatureBase<SpaceDim>::initialize(aMesh, aInputParams);
            this->initialize(aMesh, aInputParams);
            if( dm != nullptr )
            {
                dm->scalarMultiVectors[aModelName] = this->mCubWeights;
            }
        }
    }

    /******************************************************************************/
    ~CogentCubature()
    /******************************************************************************/
    {
    }
        

private:
    /********************************************************************************/
    void initialize(
      Omega_h::Mesh& aMesh,
      Teuchos::ParameterList& aInputParams)
    /********************************************************************************/
    {
        auto tCubWeights = Kokkos::create_mirror(this->mCubWeights);
        auto tWeights = this->mCogentWeights;
        auto tNumElems = tWeights.extent(0);
        auto tNumCogPtsPerElem = this->mNumCogPtsPerElem;
        for( int elemIndex=0; elemIndex<tNumElems; elemIndex++)
        {
            for( int ptIndex=0; ptIndex<tNumCogPtsPerElem; ptIndex++)
            {
                tCubWeights(elemIndex, ptIndex) = tWeights(elemIndex, ptIndex);
            }
        }
        Kokkos::deep_copy(this->mCubWeights, tCubWeights);
    }
};

} // namespace Plato

#ifdef PLATO_1D
extern template class Plato::CogentCubature<1>;
extern template class Plato::CogentCubatureDegreeOne<1>;
#endif

#ifdef PLATO_2D
extern template class Plato::CogentCubature<2>;
extern template class Plato::CogentCubatureDegreeOne<2>;
#endif

#ifdef PLATO_3D
extern template class Plato::CogentCubature<3>;
extern template class Plato::CogentCubatureDegreeOne<3>;
#endif

#endif /* COGENT_CUBATURE_HPP_ */
