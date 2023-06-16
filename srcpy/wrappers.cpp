/**
 * @file python_bindings.cpp
 * license License BSD-3-Clause
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft.
 */

#include <pybind11/pybind11.h>
#include "package_template/pid.hpp"

using namespace package_template;

PYBIND11_MODULE(package_template_cpp_bindings, m)
{
    pybind11::class_<PID>(m, "PID")
        .def(pybind11::init<>())
        .def("compute", &PID::compute)
        .def("reset_integral", &PID::reset_integral);
}
