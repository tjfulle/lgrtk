/*
 * LinearTetCubRuleDegreeOne.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef LINEARTETCUBRULEDEGREEONE_HPP_
#define LINEARTETCUBRULEDEGREEONE_HPP_

#include <string>

#include "plato/PlatoStaticsTypes.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/CubatureRule.hpp"

namespace Plato
{

/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class LinearTetCubRuleDegreeOne : public Plato::CubatureRule<SpaceDim>
/******************************************************************************/
{
public:
    /******************************************************************************/
    LinearTetCubRuleDegreeOne() : 
      Plato::CubatureRule<SpaceDim>(),
      mCubWeight(1.0)
    /******************************************************************************/
    {
        this->initialize();
    }
    /******************************************************************************/
    LinearTetCubRuleDegreeOne(Omega_h::Mesh& aMesh) : 
      Plato::CubatureRule<SpaceDim>(aMesh),
      mCubWeight(1.0)
    /******************************************************************************/
    {
        this->initialize();
    }
    /******************************************************************************/
    ~LinearTetCubRuleDegreeOne()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    KOKKOS_INLINE_FUNCTION Plato::Scalar getCubWeight() const
    /******************************************************************************/
    {
        return (mCubWeight);
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
    void initialize()
    {
        // set gauss weight
        for(Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
        { 
            mCubWeight /= Plato::Scalar(tDimIndex);
        }
        Plato::fill(mCubWeight, this->mCubWeights);
    }

    Plato::Scalar mCubWeight;
};
// class LinearTetCubRuleDegreeOne

} // namespace Plato

#endif /* LINEARTETCUBRULEDEGREEONE_HPP_ */
