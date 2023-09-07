#ifndef __force_observer_dam_tau__
#define __force_observer_dam_tau__


#include <stdexcept>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
// #include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "sobec/crocomplements/contact/multiple-contacts.hpp"
#include "sobec/fwd.hpp"

namespace mim {
namespace estimator {
/**
 * @brief Differential action model for contact forward dynamics in multibody
 * systems with torque model mismatch
 *
 * This class is derived from
 * `sobec::DifferentialActionModelContactFwdDynamicsTpl` with the additional
 * feature that it allows to propagate the estimated model mismatch delta_tau
 * through the forward dynamics model
 *
 */
class DifferentialActionModelContactFwdDynamics
    : public sobec::DifferentialActionModelContactFwdDynamicsTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef sobec::DifferentialActionModelContactFwdDynamicsTpl<double> Base;
  typedef sobec::DifferentialActionDataContactFwdDynamicsTpl<double> Data;
  typedef crocoddyl::MathBaseTpl<double> MathBase;
  typedef crocoddyl::CostModelSumTpl<double> CostModelSum;
  typedef crocoddyl::StateMultibodyTpl<double> StateMultibody;
  typedef crocoddyl::ContactModelMultipleTpl<double> crocoContactModelMultiple;
  typedef crocoddyl::ActuationModelAbstractTpl<double> ActuationModelAbstract;
  typedef crocoddyl::DifferentialActionDataAbstractTpl<double>
      DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact forward-dynamics action model
   *
   * It describes the dynamics evolution of a multibody system under
   * rigid-contact constraints defined by `ContactModelMultipleTpl`. It computes
   * the cost described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] contacts         Stack of rigid contact
   * @param[in] costs            Stack of cost functions
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia
   * matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force
   * derivatives (default false)
   */
  DifferentialActionModelContactFwdDynamics(
      boost::shared_ptr<StateMultibody> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<crocoContactModelMultiple> contacts,
      boost::shared_ptr<CostModelSum> costs,
      const double JMinvJt_damping = 0.,
      const bool enable_force = false);
  virtual ~DifferentialActionModelContactFwdDynamics();

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  void set_delta_tau(VectorXs& inDeltaTau) { delta_tau_ = inDeltaTau; } ;

 private:
  bool enable_force_;
  boost::shared_ptr<sobec::ContactModelMultipleTpl<double>> sobec_contacts_;
  VectorXs delta_tau_; 
};

}  // namespace estimator
}  // namespace mim


#endif  // __force_observer_dam_tau__