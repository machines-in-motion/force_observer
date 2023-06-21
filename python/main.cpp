#include "force_observer/python.hpp"

BOOST_PYTHON_MODULE(force_observer_pywrap) { 
    gepetto::example::exposeExampleAdder(); 
    mim::estimator::exposeEstimator(); 
}
