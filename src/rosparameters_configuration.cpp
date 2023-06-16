/**
 * @file rosparameters_configuration.cpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#include "package_template/rosparameters_configuration.hpp"

namespace package_template
{
static bool get_parameter(const ros::NodeHandle &nh,
                          const std::string &parameter,
                          double &get_value)
{
    ros::Rate wait(10);
    bool success = false;
    while (ros::ok())
    {
        success = nh.getParam(parameter, get_value);
        if (success) return true;
        wait.sleep();
    }
    return false;
}

RosParameters_configuration::RosParameters_configuration()
{
    this->error_ = false;
    this->error_message_ = "no error";
    ros::NodeHandle nh;
    std::vector<std::string> parameters = {
        ROSPARAM_KP, ROSPARAM_KD, ROSPARAM_KI};
    std::vector<double *> gains = {&(this->kp_), &(this->kd_), &(this->ki_)};
    for (unsigned int i = 0; i < parameters.size(); i++)
    {
        bool success = get_parameter(nh, parameters[i], *(gains[i]));
        if (!success)
        {
            this->error_ = true;
            this->error_message_ = "roscore shut down before parameter " +
                                   parameters[i] + " could be read";
        }
    }
}

double RosParameters_configuration::get_kp() const
{
    return this->kp_;
}

double RosParameters_configuration::get_kd() const
{
    return this->kd_;
}

double RosParameters_configuration::get_ki() const
{
    return this->ki_;
}

bool RosParameters_configuration::has_error() const
{
    return this->error_;
}

std::string RosParameters_configuration::get_error() const
{
    return this->error_message_;
}

}  // namespace package_template
