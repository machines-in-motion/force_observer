#ifndef __force_observer_python__
#define __force_observer_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "force_observer/gepadd.hpp"
#include "force_observer/estimator.hpp"

namespace gepetto {
namespace example {
void exposeExampleAdder();
}  // namespace example
}  // namespace gepetto

namespace mim{
namespace estimator{
    void exposeEstimator();
} // namespace mim
} // namespace estimator

#endif
