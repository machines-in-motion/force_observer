/**
 * @file default_configuration.hpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#pragma once

#include "package_template/gains_configuration.hpp"

#define DEFAULT_KP 1.0
#define DEFAULT_KD 1.0
#define DEFAULT_KI 1.0

namespace package_template
{
/** @brief Default configuration for the kp, kd, ki paramters.
 *
 * This class initialize the PID gains as follow:
 *  - kp = DEFAULT_KP,
 *  - kd = DEFAULT_KD
 *  - ki = DEFAULT_KI
 */
class DefaultConfiguration : public Gains_configuration
{
public:
    /** @brief Here we use the default destructor. */
    ~DefaultConfiguration()
    {
    }

    /**
     * @brief Always returns DEFAULT_KP.
     *
     * @return double DEFAULT_KP
     */
    double get_kp() const;

    /**
     * @brief Always returns DEFAULT_KD.
     *
     * @return double DEFAULT_KD
     */
    double get_kd() const;

    /**
     * @brief Always returns DEFAULT_KI.
     *
     * @return double DEFAULT_KI
     */
    double get_ki() const;

    /**
     * @brief Always returns false.
     *
     * @return true Never
     * @return false Always
     */
    bool has_error() const;

    /**
     * @brief Always returns "no error".
     *
     * @return std::string "no error"
     */
    std::string get_error() const;
};

}  // namespace package_template
