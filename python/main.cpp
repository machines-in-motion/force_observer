#include "force_observer/python.hpp"

BOOST_PYTHON_MODULE(force_observer_pywrap) { 

    namespace bp = boost::python;

    bp::import("sobec");

    mim::estimator::exposeEstimator(); 
    mim::estimator::exposeEstimatorTau(); 
    mim::estimator::exposeDAMTau(); 
    mim::estimator::exposeMHEstimator(); 
    mim::estimator::exposeMHVaryingEstimator(); 
}
