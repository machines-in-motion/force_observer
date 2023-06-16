/**
 * @file gains_configuration.cpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#include "package_template/gains_configuration.hpp"

namespace package_template
{
void print_configuration(const Gains_configuration& configuration)
{
    std::cout << "kp: " << configuration.get_kp() << std::endl;
    std::cout << "kd: " << configuration.get_kd() << std::endl;
    std::cout << "ki: " << configuration.get_ki() << std::endl;
}

}  // namespace package_template
