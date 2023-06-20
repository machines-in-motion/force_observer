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
      "This residual function is defined as r = f-fref, where f,fref describe "
      "the current and reference\n"
      "the spatial forces, respectively.",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>,
               pinocchio::FrameIndex, pinocchio::Force, std::size_t,
               std::size_t>(
          bp::args("self", "state", "id", "fref", "nc", "nu"),
          "Initialize the contact force residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: reference spatial contact force in the contact "
          "coordinates\n"
          ":param nc: dimension of the contact force (nc <= 6)\n"
          ":param nu: dimension of control vector"));

  bp::register_ptr_to_python<
      boost::shared_ptr<ForceEstimatorData> >();

  bp::class_<ForceEstimatorData>(
      "ForceEstimatorData", "Data for force estimation.\n\n",
      bp::init<ForceEstimator*>(
          bp::args("self", "model", "data"),
          "Create Center of friction residual data.\n\n"
          ":param model: residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()]);
}

}  // namespace mim
}  // namespace estimator
