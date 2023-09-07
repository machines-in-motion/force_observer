
#include "force_observer/python.hpp"
#include "force_observer/dam_tau.hpp"
// #include "crocoddyl/fwd.hpp" 
// #include "sobec/fwd.hpp"
// #include <eigenpy/eigenpy.hpp>

namespace mim {
namespace estimator {

namespace bp = boost::python;

void exposeDAMTau() {
  bp::register_ptr_to_python<boost::shared_ptr< DAMContactDeltaTau> >();

  bp::class_<DAMContactDeltaTau,
             bp::bases<sobec::newcontacts::DifferentialActionModelContactFwdDynamics> >(
      "DAMContactDeltaTau",
      "Differential action model for contact forward dynamics in multibody "
      "systems.\n\n"
      "The contact is modelled as holonomic constraits in the contact frame. "
      "There\n"
      "is also a custom implementation in case of system with armatures. If "
      "you want to\n"
      "include the armature, you need to use set_armature(). On the other "
      "hand, the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<boost::shared_ptr<crocoddyl::StateMultibody>,
               boost::shared_ptr<crocoddyl::ActuationModelAbstract>,
               boost::shared_ptr<crocoddyl::ContactModelMultiple>,
               boost::shared_ptr<crocoddyl::CostModelSum>,
               bp::optional<double, bool> >(
          bp::args("self", "state", "actuation", "contacts", "costs",
                   "inv_damping", "enable_force"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not "
          "full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost "
          "based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. "
          "enable_force=True)."
          ":param state: multibody state\n"
          ":param actuation: actuation model\n"
          ":param contacts: multiple contact model\n"
          ":param costs: stack of cost functions\n"
          ":param inv_damping: Damping factor for cholesky decomposition of "
          "JMinvJt (default 0.)\n"
          ":param enable_force: Enable the computation of force Jacobians "
          "(default False)"))
.add_property("delta_tau", bp::make_function(&DAMContactDeltaTau::get_delta_tau, bp::return_value_policy<bp::copy_const_reference>()), bp::make_function(&DAMContactDeltaTau::set_delta_tau), "delta_tau");
}

}  // namespace estimator
}  // namespace mim
