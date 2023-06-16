/**
 * @file gains_configuration.hpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#pragma once

#include <iostream>
#include <string>

namespace package_template
{
/** @brief Abstract class defining for the PID configuration.
 *
 * This virtual object describes the configuration a PID objects is waiting
 * for. Daughter class will for example be initialize through files, ROS
 * params, etc.
 */
class Gains_configuration
{
public:
    /**
     * @brief The default destructor do nothing.
     */
    virtual ~Gains_configuration()
    {
    }

    /** @brief Get the proportional gain.
     * @return double
     */
    virtual double get_kp() const = 0;

    /** @brief Get the derivative gain.
     * @return double
     */
    virtual double get_kd() const = 0;

    /** @brief Get the integral gain.
     * @return double
     */
    virtual double get_ki() const = 0;

    /** @brief Enquire if an error was encountered while reading the
     * configuration.
     * @see get_error()
     * @return true if an error has been encountered
     * @return false otherwise
     */
    virtual bool has_error() const = 0;

    /**
     * returns error encountered when reading configuration
     * @see has_error()
     */
    virtual std::string get_error() const = 0;
};

/*! print values encapsulated by the provided configuration console on the
 * standard output */
void print_configuration(const Gains_configuration& configuration);

}  // namespace package_template
