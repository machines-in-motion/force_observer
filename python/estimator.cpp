#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include "force_observer/python.hpp"
#include "force_observer/estimator.hpp"
#include <eigenpy/eigenpy.hpp>

namespace mim {
namespace estimator {

namespace bp = boost::python;


void exposeEstimator() {
  bp::register_ptr_to_python<
      boost::shared_ptr<ForceEstimator> >();

  bp::class_<ForceEstimator>(
      "ForceEstimator",
      "EStimates the contact force offset from prior & force/state measurements.",
      bp::init<boost::shared_ptr<pinocchio::Model>,
               std::size_t,
               std::size_t,
               const pinocchio::FrameIndex, 
               const Eigen::Vector2d&, 
               const pinocchio::ReferenceFrame>(
          bp::args("self", "pin_model", "nc", "nc_delta_f", "frameId", "baumgarte_gains", "ref"),
          "Initialize contact force offset estimator.\n\n"
          ":param pin_model: Pinocchio model\n"
          ":param nc: Dimension of the contact model\n"
          ":param nc_delta_f: Dimension of the force offset\n"
          ":param frameId: Contact frame id\n"
          ":param baumgarte_gains: Baumgarte gains in contact model\n"
          ":param ref: Pinocchio reference frame of the contact model."))

      .def<void (ForceEstimator::*)(const boost::shared_ptr<ForceEstimatorData>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ForceEstimator::estimate, bp::args("self", "data", "q", "v", "a", "tau", "df_prior", "F_mes"),
          "Computes the force offset estimate.\n\n"
          ":param data: force estimator data\n"
          ":param q: joint position vector\n"
          ":param v: joint velocity vector\n"
          ":param a: joint acceleration vector\n"
          ":param tau: joint torque vector\n"
          ":param df_prior: contact force offset prior\n"
          ":param F_mes: measured contact force")

      .add_property(
          "pinocchio",
          bp::make_function(&ForceEstimator::get_pinocchio, bp::return_internal_reference<>()),
          "multibody model (i.e. pinocchio model)");

  bp::register_ptr_to_python<
      boost::shared_ptr<ForceEstimatorData> >();

  bp::class_<ForceEstimatorData>(
      "ForceEstimatorData", "Data for force estimation.\n\n",
      bp::init<ForceEstimator*>(
          bp::args("self", "model"),
          "Create force estimator data.\n\n"
          ":param model: force estimator model.")) 
        //   [bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])

      .add_property("pinocchio", bp::make_getter(&ForceEstimatorData::pinocchio, bp::return_internal_reference<>()), "pinocchio data")
      .add_property("F", bp::make_getter(&ForceEstimatorData::F, bp::return_internal_reference<>()), "measured contact force");
}

}  // namespace mim
}  // namespace estimator
