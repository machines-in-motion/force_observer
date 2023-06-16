/**
 * @file pid.cpp
 * @author Vincent Berenz
 * @copyright Copyright (c) 2019, New York University and Max Planck
 * Gesellschaft, License BSD-3-Clause
 * @date 2019-12-09
 */

#include "package_template/pid.hpp"

namespace package_template
{
PID::PID() : integral_(0)
{
    configuration_ = new DefaultConfiguration();
    private_configuration_ = true;
}

PID::PID(const Gains_configuration& configuration)
    : configuration_(&configuration),
      private_configuration_(false),
      integral_(0)
{
}

PID::~PID()
{
    if (private_configuration_)
    {
        delete configuration_;
    }
}

double PID::compute(const double position,
                    const double velocity,
                    const double position_target,
                    const double delta_time)
{
    double position_error = position_target - position;
    integral_ += delta_time * position_error;
    double f = position_error * configuration_->get_kp() -
               velocity * configuration_->get_kd() +
               integral_ * configuration_->get_ki();
    return f;
}

void PID::reset_integral()
{
    this->integral_ = 0;
}

/** @brief Use a PID factory for the unittests. */
class Default_pid_factory
{
public:
    /** The PID gains. */
    static std::vector<std::shared_ptr<Gains_configuration> > configs_;
    /** List of PID controllers. */
    static std::vector<std::shared_ptr<PID> > controllers_;
    /**
     * @brief PID controller factory.
     *
     * @return PID& Return a reference to a newly created PID controller.
     */
    static PID& get()
    {
        std::shared_ptr<Gains_configuration> configuration(
            new DefaultConfiguration());
        std::shared_ptr<PID> controller(new PID(*configuration));
        configs_.push_back(configuration);
        controllers_.push_back(controller);
        return *controller;
    }
};

std::vector<std::shared_ptr<Gains_configuration> >
    Default_pid_factory::configs_;
std::vector<std::shared_ptr<PID> > Default_pid_factory::controllers_;

PID& get_default_pid()
{
    return Default_pid_factory::get();
}

}  // namespace package_template
