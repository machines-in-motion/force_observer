/**
 * @file file_configuration.cpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#include "package_template/file_configuration.hpp"

namespace package_template
{
File_configuration::File_configuration(std::string yaml_file)
{
    try
    {
        YAML::Node node = YAML::LoadFile(yaml_file);
        this->kp_ = node["kp"].as<double>();
        this->kd_ = node["kd"].as<double>();
        this->ki_ = node["ki"].as<double>();

        this->error_ = false;
    }
    catch (const std::exception& e)
    {
        this->error_ = true;
        this->error_message_ = e.what();
    }
}

double File_configuration::get_kp() const
{
    return this->kp_;
}

double File_configuration::get_kd() const
{
    return this->kd_;
}

double File_configuration::get_ki() const
{
    return this->ki_;
}

bool File_configuration::has_error() const
{
    return this->error_;
}

std::string File_configuration::get_error() const
{
    return this->error_message_;
}

}  // namespace package_template
