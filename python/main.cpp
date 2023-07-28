#include "force_observer/python.hpp"

BOOST_PYTHON_MODULE(force_observer_pywrap) { 
    mim::estimator::exposeEstimator(); 
    mim::estimator::exposeMHEstimator(); 
}
