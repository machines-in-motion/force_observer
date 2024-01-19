#include "force_observer/python.hpp"
#include "force_observer/mh_varying_estimator.hpp"

#include <eigenpy/eigenpy.hpp>

namespace mim {
namespace estimator {

namespace bp = boost::python;


void exposeMHVaryingEstimator() {
  bp::register_ptr_to_python<
      boost::shared_ptr<MHVaryingForceEstimator> >();

  bp::class_<MHVaryingForceEstimator>(
      "MHVaryingForceEstimator",
      "EStimates the contact force offset from prior & force/state measurements.",
      bp::init<std::size_t,
               pinocchio::Model&,
               std::size_t,
               const pinocchio::FrameIndex, 
               const Eigen::Vector2d&, 
               const pinocchio::ReferenceFrame>(
          bp::args("self", "T", "pin_model", "nc", "frameId", "baumgarte_gains", "ref"),
          "Initialize contact force offset estimator.\n\n"
          ":param T: Horizon size\n"
          ":param pin_model: Pinocchio model\n"
          ":param nc: Dimension of the contact model\n"
          ":param frameId: Contact frame id\n"
          ":param baumgarte_gains: Baumgarte gains in contact model\n"
          ":param ref: Pinocchio reference frame of the contact model."))

      .def<void (MHVaryingForceEstimator::*)(const boost::shared_ptr<MHVaryingForceEstimatorData>&,
                                    std::vector<Eigen::VectorXd>,
                                    std::vector<Eigen::VectorXd>,
                                    std::vector<Eigen::VectorXd>,
                                    std::vector<Eigen::VectorXd>,
                                    const Eigen::Ref<const Eigen::VectorXd>&,
                                    std::vector<Eigen::VectorXd>)>(
          "estimate", &MHVaryingForceEstimator::estimate, bp::args("self", "data", "q_list", "v_list", "a_list", "tau_list", "df_prior", "F_mes_list"),
          "Computes the force offset estimate.\n\n"
          ":param data: force estimator data\n"
          ":param q_list: joint position vector list\n"
          ":param v_list: joint velocity vector list\n"
          ":param a_list: joint acceleration vector list\n"
          ":param tau_list: joint torque vector list\n"
          ":param df_prior: contact force offset prior\n"
          ":param F_mes_list: measured contact force list")

      .def("createData", &MHVaryingForceEstimator::createData,
           bp::args("self"), "Create the Force estimator data.")

      .add_property("pinocchio", bp::make_function(&MHVaryingForceEstimator::get_pinocchio, bp::return_value_policy<bp::return_by_value>()), "multibody model (i.e. pinocchio model)")
      .add_property("nv", bp::make_function(&MHVaryingForceEstimator::get_nv, bp::return_value_policy<bp::return_by_value>()), "Size of the joint velocity vector")
      .add_property("nc", bp::make_function(&MHVaryingForceEstimator::get_nc, bp::return_value_policy<bp::return_by_value>()), "Size of the contact model")
      .add_property("frame_id", bp::make_function(&MHVaryingForceEstimator::get_frameId, bp::return_value_policy<bp::return_by_value>()), "Frame id of the contact frame")
      .add_property("baumgarte_gains", bp::make_function(&MHVaryingForceEstimator::get_baumgarte_gains, bp::return_value_policy<bp::return_by_value>()), "Baumgarte gains of the contact model")
      .add_property("ref", bp::make_function(&MHVaryingForceEstimator::get_ref, bp::return_value_policy<bp::return_by_value>()), "Pinocchio reference frame of the contact model")
      .add_property("n_tot", bp::make_function(&MHVaryingForceEstimator::get_n_tot, bp::return_value_policy<bp::return_by_value>()), "Total size of the estimation QP")
      .add_property("n_eq", bp::make_function(&MHVaryingForceEstimator::get_neq, bp::return_value_policy<bp::return_by_value>()), "Number of equality constraints of the estimation QP")
      .add_property("n_in", bp::make_function(&MHVaryingForceEstimator::get_nin, bp::return_value_policy<bp::return_by_value>()), "Number of inequality constraints of the estimation QP")
      .add_property("P", bp::make_function(&MHVaryingForceEstimator::get_P, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&MHVaryingForceEstimator::set_P), "QP parameter")
      .add_property("Q", bp::make_function(&MHVaryingForceEstimator::get_Q, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&MHVaryingForceEstimator::set_Q), "QP parameter")
      .add_property("R", bp::make_function(&MHVaryingForceEstimator::get_R, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&MHVaryingForceEstimator::set_R), "QP parameter")
      .add_property("H", bp::make_function(&MHVaryingForceEstimator::get_H, bp::return_value_policy<bp::return_by_value>()), "QP parameter")
      .add_property("mask", bp::make_function(&MHVaryingForceEstimator::get_mask), bp::make_function(&MHVaryingForceEstimator::set_mask), "Contact model mask (for 1D only)");

  bp::register_ptr_to_python<
      boost::shared_ptr<MHVaryingForceEstimatorData> >();

  bp::class_<MHVaryingForceEstimatorData>(
      "MHVaryingForceEstimatorData", "Data for force estimation.\n\n",
      bp::init<MHVaryingForceEstimator*>(
          bp::args("self", "model"),
          "Create force estimator data.\n\n"
          ":param model: force estimator model.")) 

      .add_property("pinocchio", bp::make_getter(&MHVaryingForceEstimatorData::pinocchio, bp::return_internal_reference<>()), "pinocchio data")
      .add_property("F", bp::make_getter(&MHVaryingForceEstimatorData::F, bp::return_value_policy<bp::return_by_value>()), "measured contact force")
      .add_property("delta_f", bp::make_getter(&MHVaryingForceEstimatorData::delta_f, bp::return_value_policy<bp::return_by_value>()), "contact force offset estimate")
      .add_property("J", bp::make_getter(&MHVaryingForceEstimatorData::J, bp::return_value_policy<bp::return_by_value>()), "full contact Jacobian")
      .add_property("J1", bp::make_getter(&MHVaryingForceEstimatorData::J1, bp::return_value_policy<bp::return_by_value>()), "Jacobian 1")
      .add_property("alpha0", bp::make_getter(&MHVaryingForceEstimatorData::alpha0, bp::return_value_policy<bp::return_by_value>()), "Contact acceleration drift")
      .add_property("nu", bp::make_getter(&MHVaryingForceEstimatorData::nu, bp::return_value_policy<bp::return_by_value>()), "Contact velocity")
      .add_property("M", bp::make_getter(&MHVaryingForceEstimatorData::M, bp::return_value_policy<bp::return_by_value>()), "Generalized inertia matrix")
      .add_property("h", bp::make_getter(&MHVaryingForceEstimatorData::h, bp::return_value_policy<bp::return_by_value>()), "Nonlinear terms of RNEA")
      .add_property("b", bp::make_getter(&MHVaryingForceEstimatorData::b, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("A", bp::make_getter(&MHVaryingForceEstimatorData::A, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("g", bp::make_getter(&MHVaryingForceEstimatorData::g, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("C", bp::make_getter(&MHVaryingForceEstimatorData::C, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("l", bp::make_getter(&MHVaryingForceEstimatorData::l, bp::return_value_policy<bp::return_by_value>()), "QP param")
      .add_property("u", bp::make_getter(&MHVaryingForceEstimatorData::u, bp::return_value_policy<bp::return_by_value>()), "QP param");
}

}  // namespace mim
}  // namespace estimator
