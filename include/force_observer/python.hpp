#ifndef __force_observer_python__
#define __force_observer_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "force_observer/estimator.hpp"
#include "force_observer/estimator_tau.hpp"
#include "force_observer/dam_tau.hpp"
#include "force_observer/mh_estimator.hpp"
#include "force_observer/mh_varying_estimator.hpp"


namespace mim{
namespace estimator{
    void exposeEstimator();
    void exposeEstimatorTau();
    void exposeDAMTau();
    void exposeMHEstimator();
    void exposeMHVaryingEstimator();
} // namespace mim
} // namespace estimator

#endif
