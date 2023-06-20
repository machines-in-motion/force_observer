#include "force_observer/estimator.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace mim {
namespace estimator {


ForceEstimator::ForceEstimator(
      boost::shared_ptr<pinocchio::Model> pin_model,
      boost::shared_ptr<pinocchio::Data> pin_data,
      std::size_t nc,
      std::size_t nc_delta_f,
      const pinocchio::FrameIndex frameId,
      const Vector2d& baumgarte_gains, 
      const pinocchio::ReferenceFrame ref
    ) : pin_model_(pin_model),
        pin_data_(pin_data),
        nc_(nc),
        nc_delta_f_(nc_delta_f),
        frameId_(frameId),
        baumgarte_gains_(baumgarte_gains),
        ref_(ref)    
        {
    // Dimensions
    nv_ = pin_model->get_nv();
    n_tot_ = nv_ + nc_ + nc_delta_f_;
    neq_ = nv_ + nc_;
    nin_ = 0;

    // Default weights
    P_ = 1e-0*Eigen::VectorXd::Ones(nc_delta_f_);
    Q_ = 1e-2*Eigen::VectorXd::Ones(nv_);
    R_ = 1e-2*Eigen::VectorXd::Ones(nc_);
    H_ = Eigen::MatrixXd::Zero(n_tot_);
    H_.bottomRightCorner(nc_delta_f_, nc_delta_f_) = P_.asDiagonal();
    H_.topLeftCorner(n_tot_, n_tot_) = Q_.asDiagonal();
    H_.block(nv_, nv_, nc_, nc_) = R_.asDiagonal();

    if(baumgarte_gains_[0] > 1e-6){
        std::cout << "Error: the proportional gain of Baugmarte should be 0 !" << std::endl;
    }
    std::cout << "Initialized force estimator." << std::endl;
}

ForceEstimator::~ForceEstimator(){}

void ForceEstimator::estimate(
                const boost::shared_ptr<ForceEstimatorData>& data, 
                const Eigen::Ref<const Eigen::VectorXd>& q,
                const Eigen::Ref<const Eigen::VectorXd>& v,
                const Eigen::Ref<const Eigen::VectorXd>& a,
                const Eigen::Ref<const Eigen::VectorXd>& tau,
                const Eigen::Ref<const Eigen::VectorXd>& df_prior,
                const Eigen::Ref<const Eigen::VectorXd>& F_mes){
    Data* d = static_cast<Data*>(data.get());
    
    // Compute required dynamic quantities
    pinocchio::computeAllTerms(pin_model_, pin_data_, q, v);
    pinocchio::forwardKinematics(pin_model_, pin_data_, q, v, a);
    pinocchio::updateFramePlacements(pin_model_, pin_data_);
    d->h = pin_data_->nle;
    d->M = pin_data_->M;

    if(nc_ == 1){
        d->alpha0 = pinocchio::getFrameAcceleration(pin_model_, pin_data_, frameId_).vector()[mask_];
        d->nu = pinocchio::getFrameVelocity(pin_model_, pin_data_, frameId_).vector()[mask_];
        d->J1 = pinocchio::getFrameJacobian(pin_model_, pin_data_, frameId_, ref_).row(mask_);
    } else {
        d->alpha0 = pinocchio::getFrameAcceleration(pin_model_, pin_data_, frameId_).vector().head(nc_);
        d->nu = pinocchio::getFrameVelocity(pin_model_, pin_data_, frameId_).vector().head(nc_);
        d->J1 = pinocchio::getFrameJacobian(pin_model_, pin_data_, frameId_, ref_).topRows(nc_);
    }
    d->alpha0 -= baumgarte_gains_[1] * d->nu;
    if(nc_delta_f_ == 3 && nc_ == 1):
        d->J2 = pinocchio::getFrameJacobian(pin_model_, pin_data_, frameId_, ref_).topRows(nc_delta_f_);
    else:
        d->J2 = d->J1;
    
    // Construct QP
    d->b.topRows(nv_) = d->h - tau; 
    d->b.bottomRows(nc_) = -d->alpha0;
    d->A.topLeftCorner(nv_, nv_) = d->M;
    d->A.block(0, nv_, nv_, nc_) = d->J1.transpose();
    d->A.block(0, nv_+nc_, nv_, nc_delta_f_) = d->J2.transpose();
    d->A.block(nv, 0, nv_+nv_, nc_) = d->J1;

    d->g.head(nv_) = -Q_ * a
    d->g.segment(nv_, nc_) = -R_ * F_mes
    d->g.tail(nc_delta_f_)= - P_ * df_prior
    
    qp_.init(H_, d->g, d->A, d->b); //, self.C, self.l, self.u)

    // t1 = time.time()
    qp_.solve();
}

std::size_t ForceEstimator::get_nv()const {
    return nv_;
}

std::size_t ForceEstimator::get_nc()const {
    return nc_;
}

std::size_t ForceEstimator::get_nc_delta_f() const {
    return nc_delta_f_;
}

const Eigen::Vector2d& ForceEstimator::get_baumgarte_gains() const {
    return baumgarte_gains_;
}

std::size_t ForceEstimator::get_frameId() const {
    return frameId_;
}

pinocchio::ReferenceFrame ForceEstimator::get_ref() const {
    return ref_;
}

std::size_t ForceEstimator::get_n_tot()const {
    return n_tot_;
}

const Eigen::VectorXd& ForceEstimator::get_P()const {
    return P_;
}

const Eigen::VectorXd& ForceEstimator::get_Q()const {
    return Q_;
}

const Eigen::VectorXd& ForceEstimator::get_R()const {
    return R_;
}

const Eigen::MatrixXd& ForceEstimator::get_H()const {
    return H_;
}

void ForceEstimator::set_nc(const std::size_t nc) {
    nc_ = nc;
}

void ForceEstimator::set_nc_delta_f(const std::size_t nc_delta_f) {
    nc_delta_f_ = nc_delta_f;
}

void ForceEstimator::set_baumgarte_gains(const Eigen::Vector2d& gains) {
    baumgarte_gains_ = gains;
}

void ForceEstimator::set_frameId(const std::size_t frameId) {
    frameId_ = frameId;
}

void ForceEstimator::set_ref(const pinocchio::ReferenceFrame ref) {
    ref_ = ref;
}

void ForceEstimator::set_ref(const Eigen::VectorXd& inP) {
    P_ = inP; 
}

void ForceEstimator::set_ref(const Eigen::VectorXd& inQ) {
    Q_ = inQ;
}

void ForceEstimator::set_ref(const Eigen::VectorXd& inR) {
    Q_ = inR;
}

boost::shared_ptr<ForceEstimatorData> ForceEstimator::createData() {
  return boost::allocate_shared<ForceEstimatorData>(Eigen::aligned_allocator<ForceEstimatorData>(), this);
}

}  // namespace mim
}  // namespace estimator
