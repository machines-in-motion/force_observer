#include "force_observer/python.hpp"
#include "force_observer/estimator_tau.hpp"

namespace mim {
namespace estimator {

namespace bp = boost::python;


void exposeEstimatorTau() {
  bp::register_ptr_to_python<
      boost::shared_ptr<TorqueEstimator> >();

  bp::class_<TorqueEstimator>(
      "TorqueEstimator",
      "EStimates the contact force offset from prior & force/state measurements.",
      bp::init<pinocchio::Model&,
               std::size_t,
               const pinocchio::FrameIndex, 
               const Eigen::Vector2d&, 
               const pinocchio::ReferenceFrame>(
          bp::args("self", "pin_model", "nc", "frameId", "baumgarte_gains", "ref"),
          "Initialize contact force offset estimator.\n\n"
          ":param pin_model: Pinocchio model\n"
          ":param nc: Dimension of the contact model\n"
          ":param frameId: Contact frame id\n"
          ":param baumgarte_gains: Baumgarte gains in contact model\n"
          ":param ref: Pinocchio reference frame of the contact model."))

      .def<void (TorqueEstimator::*)(const boost::shared_ptr<TorqueEstimatorData>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    const Eigen::Ref<const Eigen::VectorXd>&)>(
          "estimate", &TorqueEstimator::estimate, bp::args("self", "data", "q", "v", "a", "tau", "dtau_prior", "F_mes"),
          "Computes the force offset estimate.\n\n"
          ":param data: force estimator data\n"
          ":param q: joint position vector\n"
          ":param v: joint velocity vector\n"
          ":param a: joint acceleration vector\n"
          ":param tau: joint torque vector\n"
          ":param dtau_prior: contact force offset prior\n"
          ":param F_mes: measured contact force")

      .def("createData", &TorqueEstimator::createData,
           bp::args("self"), "Create the Force estimator data.")

      .add_property("pinocchio", bp::make_function(&TorqueEstimator::get_pinocchio, bp::return_value_policy<bp::return_by_value>()), "multibody model (i.e. pinocchio model)")
      .add_property("nv", bp::make_function(&TorqueEstimator::get_nv, bp::return_value_policy<bp::return_by_value>()), "Size of the joint velocity vector")
      .add_property("nc", bp::make_function(&TorqueEstimator::get_nc, bp::return_value_policy<bp::return_by_value>()), "Size of the contact model")
      .add_property("frame_id", bp::make_function(&TorqueEstimator::get_frameId, bp::return_value_policy<bp::return_by_value>()), "Frame id of the contact frame")
      .add_property("baumgarte_gains", bp::make_function(&TorqueEstimator::get_baumgarte_gains, bp::return_value_policy<bp::return_by_value>()), "Baumgarte gains of the contact model")
      .add_property("ref", bp::make_function(&TorqueEstimator::get_ref, bp::return_value_policy<bp::return_by_value>()), "Pinocchio reference frame of the contact model")
      .add_property("n_tot", bp::make_function(&TorqueEstimator::get_n_tot, bp::return_value_policy<bp::return_by_value>()), "Total size of the estimation QP")
      .add_property("n_eq", bp::make_function(&TorqueEstimator::get_neq, bp::return_value_policy<bp::return_by_value>()), "Number of equality constraints of the estimation QP")
      .add_property("n_in", bp::make_function(&TorqueEstimator::get_nin, bp::return_value_policy<bp::return_by_value>()), "Number of inequality constraints of the estimation QP")
      .add_property("P", bp::make_function(&TorqueEstimator::get_P, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&TorqueEstimator::set_P), "QP parameter")
      .add_property("Q", bp::make_function(&TorqueEstimator::get_Q, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&TorqueEstimator::set_Q), "QP parameter")
      .add_property("R", bp::make_function(&TorqueEstimator::get_R, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&TorqueEstimator::set_R), "QP parameter")
      .add_property("H", bp::make_function(&TorqueEstimator::get_H, bp::return_value_policy<bp::return_by_value>()), "QP parameter")
      .add_property("mask", bp::make_function(&TorqueEstimator::get_mask), bp::make_function(&TorqueEstimator::set_mask), "Contact model mask (for 1D only)");

  bp::register_ptr_to_python<
      boost::shared_ptr<TorqueEstimatorData> >();

  bp::class_<TorqueEstimatorData>(
      "TorqueEstimatorData", "Data for force estimation.\n\n",
      bp::init<TorqueEstimator*>(
          bp::args("self", "model"),
          "Create force estimator data.\n\n"
          ":param model: force estimator model.")) 
        //   [bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])

      .add_property("pinocchio", bp::make_getter(&TorqueEstimatorData::pinocchio, bp::return_internal_reference<>()), "pinocchio data")
      .add_property("F", bp::make_getter(&TorqueEstimatorData::F, bp::return_value_policy<bp::return_by_value>()), "measured contact force")
      .add_property("delta_tau", bp::make_getter(&TorqueEstimatorData::delta_tau, bp::return_value_policy<bp::return_by_value>()), "joint torque offset estimate")
      .add_property("J", bp::make_getter(&TorqueEstimatorData::J, bp::return_value_policy<bp::return_by_value>()), "full contact Jacobian")
      .add_property("J1", bp::make_getter(&TorqueEstimatorData::J1, bp::return_value_policy<bp::return_by_value>()), "Jacobian 1")
      .add_property("alpha0", bp::make_getter(&TorqueEstimatorData::alpha0, bp::return_value_policy<bp::return_by_value>()), "Contact acceleration drift")
      .add_property("nu", bp::make_getter(&TorqueEstimatorData::nu, bp::return_value_policy<bp::return_by_value>()), "Contact velocity")
      .add_property("M", bp::make_getter(&TorqueEstimatorData::M, bp::return_value_policy<bp::return_by_value>()), "Generalized inertia matrix")
      .add_property("h", bp::make_getter(&TorqueEstimatorData::h, bp::return_value_policy<bp::return_by_value>()), "Nonlinear terms of RNEA")
      .add_property("b", bp::make_getter(&TorqueEstimatorData::b, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("A", bp::make_getter(&TorqueEstimatorData::A, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("g", bp::make_getter(&TorqueEstimatorData::g, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("C", bp::make_getter(&TorqueEstimatorData::C, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("l", bp::make_getter(&TorqueEstimatorData::l, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("u", bp::make_getter(&TorqueEstimatorData::u, bp::return_value_policy<bp::return_by_value>()), "QP param");
}

}  // namespace mim
}  // namespace estimator
