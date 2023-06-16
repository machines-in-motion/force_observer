/**
 * @file default_configuration.cpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#include "package_template/default_configuration.hpp"

namespace package_template
{
double DefaultConfiguration::get_kp() const
{
    return DEFAULT_KP;
}

double DefaultConfiguration::get_kd() const
{
    return DEFAULT_KD;
}

double DefaultConfiguration::get_ki() const
{
    return DEFAULT_KI;
}

bool DefaultConfiguration::has_error() const
{
    return false;
}

std::string DefaultConfiguration::get_error() const
{
    return std::string("no error");
}

}  // namespace package_template
