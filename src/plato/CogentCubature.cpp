/*
 * CogentCubature.cpp
 *
 *  Created on: May 3, 2019
 */
#ifdef PLATO_GEOMETRY

#include "plato/CogentCubature.hpp"

#ifdef PLATO_1D
template class Plato::CogentCubature<1>;
template class Plato::CogentCubatureDegreeOne<1>;
#endif

#ifdef PLATO_2D
template class Plato::CogentCubature<2>;
template class Plato::CogentCubatureDegreeOne<2>;
#endif

#ifdef PLATO_3D
template class Plato::CogentCubature<3>;
template class Plato::CogentCubatureDegreeOne<3>;
#endif

#endif /* PLATO_GEOMETRY */
