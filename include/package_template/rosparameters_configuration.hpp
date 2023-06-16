/**
 * @file rosparameters_configuration.hpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#pragma once

#include "package_template_cpp/gains_configuration.hpp"
#include "ros/master.h"
#include "ros/ros.h"

#define ROSPARAM_KP "gains_kp"
#define ROSPARAM_KD "gains_kd"
#define ROSPARAM_KI "gains_ki"

namespace package_template
{
/*! @brief Read gains configuration from the ros parameter server. */
class RosParameters_configuration : public Gains_configuration
{
public:
    /**
     * Attempt to get the gains from the parameter server
     * ("gains_kp","gains_kd","gains_ki" parameters) If roscore is running,
     * calls to this constructor will be blocking until all the gains are read
     * or roscore is turned off. If roscore is turned off before gains are read,
     * has_error() will return true
     * @see has_error()
     */
    RosParameters_configuration();

    /*! @brief Get the proportinal gain */
    double get_kp() const;
    /*! @brief Get the derivative gain */
    double get_kd() const;
    /*! get the integral gain */
    double get_ki() const;
    /*! Check if there are internal errors */
    bool has_error() const;
    /*! Get the error messages */
    std::string get_error() const;

private:
    double kp_;                 /*!< Proportinal gain. */
    double kd_;                 /*!< Derivative gain. */
    double ki_;                 /*!< Integral gain. */
    std::string error_message_; /*!< Internal error message. */
    bool error_;                /*!< True is an error occured. */
};

}  // namespace package_template
