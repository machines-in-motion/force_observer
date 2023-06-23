#include "force_observer/python.hpp"
#include "force_observer/estimator.hpp"

namespace mim {
namespace estimator {

namespace bp = boost::python;


void exposeEstimator() {
  bp::register_ptr_to_python<
      boost::shared_ptr<ForceEstimator> >();

  bp::class_<ForceEstimator>(
      "ForceEstimator",
      "EStimates the contact force offset from prior & force/state measurements.",
      bp::init<pinocchio::Model&,
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
          "estimate", &ForceEstimator::estimate, bp::args("self", "data", "q", "v", "a", "tau", "df_prior", "F_mes"),
          "Computes the force offset estimate.\n\n"
          ":param data: force estimator data\n"
          ":param q: joint position vector\n"
          ":param v: joint velocity vector\n"
          ":param a: joint acceleration vector\n"
          ":param tau: joint torque vector\n"
          ":param df_prior: contact force offset prior\n"
          ":param F_mes: measured contact force")

      .def("createData", &ForceEstimator::createData,
           bp::args("self"), "Create the Force estimator data.")

      .add_property("pinocchio", bp::make_function(&ForceEstimator::get_pinocchio, bp::return_value_policy<bp::return_by_value>()), "multibody model (i.e. pinocchio model)")
      .add_property("nv", bp::make_function(&ForceEstimator::get_nv, bp::return_value_policy<bp::return_by_value>()), "Size of the joint velocity vector")
      .add_property("nc", bp::make_function(&ForceEstimator::get_nc, bp::return_value_policy<bp::return_by_value>()), "Size of the contact model")
      .add_property("nc_delta_f", bp::make_function(&ForceEstimator::get_nc_delta_f, bp::return_value_policy<bp::return_by_value>()), "Size of the contact force offset estimate")
      .add_property("frame_id", bp::make_function(&ForceEstimator::get_frameId, bp::return_value_policy<bp::return_by_value>()), "Frame id of the contact frame")
      .add_property("baumgarte_gains", bp::make_function(&ForceEstimator::get_baumgarte_gains, bp::return_value_policy<bp::return_by_value>()), "Baumgarte gains of the contact model")
      .add_property("ref", bp::make_function(&ForceEstimator::get_ref, bp::return_value_policy<bp::return_by_value>()), "Pinocchio reference frame of the contact model")
      .add_property("n_tot", bp::make_function(&ForceEstimator::get_n_tot, bp::return_value_policy<bp::return_by_value>()), "Total size of the estimation QP")
      .add_property("n_eq", bp::make_function(&ForceEstimator::get_neq, bp::return_value_policy<bp::return_by_value>()), "Number of equality constraints of the estimation QP")
      .add_property("n_in", bp::make_function(&ForceEstimator::get_nin, bp::return_value_policy<bp::return_by_value>()), "Number of inequality constraints of the estimation QP")
      .add_property("P", bp::make_function(&ForceEstimator::get_P, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&ForceEstimator::set_P), "QP parameter")
      .add_property("Q", bp::make_function(&ForceEstimator::get_Q, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&ForceEstimator::set_Q), "QP parameter")
      .add_property("R", bp::make_function(&ForceEstimator::get_R, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&ForceEstimator::set_R), "QP parameter")
      .add_property("H", bp::make_function(&ForceEstimator::get_H, bp::return_value_policy<bp::return_by_value>()), "QP parameter")
      .add_property("mask", bp::make_function(&ForceEstimator::get_mask), bp::make_function(&ForceEstimator::set_mask), "Contact model mask (for 1D only)");

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
      .add_property("F", bp::make_getter(&ForceEstimatorData::F, bp::return_value_policy<bp::return_by_value>()), "measured contact force")
      .add_property("delta_f", bp::make_getter(&ForceEstimatorData::delta_f, bp::return_value_policy<bp::return_by_value>()), "contact force offset estimate")
      .add_property("J", bp::make_getter(&ForceEstimatorData::J, bp::return_internal_reference<>()), "full contact Jacobian")
      .add_property("J1", bp::make_getter(&ForceEstimatorData::J1, bp::return_internal_reference<>()), "Jacobian 1")
      .add_property("J2", bp::make_getter(&ForceEstimatorData::J2, bp::return_internal_reference<>()), "Jacobian 2")
      .add_property("alpha0", bp::make_getter(&ForceEstimatorData::alpha0, bp::return_internal_reference<>()), "Contact acceleration drift")
      .add_property("nu", bp::make_getter(&ForceEstimatorData::nu, bp::return_internal_reference<>()), "Contact velocity")
      .add_property("M", bp::make_getter(&ForceEstimatorData::M, bp::return_internal_reference<>()), "Generalized inertia matrix")
      .add_property("h", bp::make_getter(&ForceEstimatorData::h, bp::return_internal_reference<>()), "Nonlinear terms of RNEA")
      .add_property("b", bp::make_getter(&ForceEstimatorData::b, bp::return_internal_reference<>()), "QP param")
      .add_property("A", bp::make_getter(&ForceEstimatorData::A, bp::return_internal_reference<>()), "QP param")
      .add_property("g", bp::make_getter(&ForceEstimatorData::g, bp::return_internal_reference<>()), "QP param")
      .add_property("C", bp::make_getter(&ForceEstimatorData::C, bp::return_internal_reference<>()), "QP param")
      .add_property("l", bp::make_getter(&ForceEstimatorData::l, bp::return_internal_reference<>()), "QP param")
      .add_property("u", bp::make_getter(&ForceEstimatorData::u, bp::return_internal_reference<>()), "QP param");
}

}  // namespace mim
}  // namespace estimator
