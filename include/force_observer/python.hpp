#ifndef __force_observer_python__
#define __force_observer_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "force_observer/estimator.hpp"
#include "force_observer/mh_estimator.hpp"


namespace mim{
namespace estimator{
    void exposeEstimator();
    void exposeMHEstimator();
} // namespace mim
} // namespace estimator

#endif
