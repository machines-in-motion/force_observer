///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/core/utils/math.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

#include "force_observer/dam_tau.hpp"

namespace mim {
namespace estimator {

DAMContactDeltaTau::
    DAMContactDeltaTau(
        boost::shared_ptr<StateMultibody> state,
        boost::shared_ptr<ActuationModelAbstract> actuation,
        boost::shared_ptr<crocoContactModelMultiple> contacts,
        boost::shared_ptr<CostModelSum> costs, const double JMinvJt_damping,
        const bool enable_force)
    : Base(state, actuation, contacts, costs, JMinvJt_damping, enable_force) ,
      enable_force_(enable_force) {
  croco_contacts_ = boost::static_pointer_cast<crocoddyl::ContactModelMultipleTpl<double>>(contacts);
  delta_tau_ = VectorXd::Zero(state->get_nv());
}

DAMContactDeltaTau::~DAMContactDeltaTau() {}

void DAMContactDeltaTau::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, 
    const Eigen::Ref<const VectorXd>& x,
    const Eigen::Ref<const VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != this->get_state()->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(this->get_state()->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != this->get_nu()) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(this->get_nu()) + ")");
  }
  const std::size_t nc = croco_contacts_->get_nc();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXd>, Eigen::Dynamic> q = x.head(this->get_state()->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXd>, Eigen::Dynamic> v = x.tail(this->get_state()->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(this->get_pinocchio(), d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(this->get_pinocchio(), d->pinocchio);

//   if (!this->get_with_armature()) {
//     d->pinocchio.M.diagonal() += this->get_armature();
//   }
  this->get_actuation()->calc(d->multibody.actuation, x, u);
  croco_contacts_->calc(d->multibody.contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXd> Jc_lu(d->multibody.contacts->Jc.topRows(nc));

  if (Jc_lu.rank() < d->multibody.contacts->Jc.topRows(nc).rows() && JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty("A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif
  pinocchio::forwardDynamics(this->get_pinocchio(), d->pinocchio, d->multibody.actuation->tau + delta_tau_,
                             d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
                             0.);
  d->xout = d->pinocchio.ddq;
  croco_contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  croco_contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);

  // Computing the cost value and residuals
  this->get_costs()->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

}  // namespace estimator
}  // namespace mim
