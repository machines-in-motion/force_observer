#include "force_observer/estimator_tau.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>

using namespace proxsuite::proxqp;

namespace mim {
namespace estimator {


TorqueEstimator::TorqueEstimator(
      pinocchio::Model& pin_model,
      std::size_t nc,
      const pinocchio::FrameIndex frameId,
      const Vector2d& baumgarte_gains, 
      const pinocchio::ReferenceFrame ref
    ) : pinocchio_(pin_model),
        nc_(nc),
        frameId_(frameId),
        baumgarte_gains_(baumgarte_gains),
        ref_(ref)    
        {
    // Dimensions
    nv_ = pinocchio_.nv;
    n_tot_ = nv_ + nc_ + nv_;
    neq_ = nv_ + nc_;
    nin_ = 0.;

    // Set default mask if 1D model
    if(nc_ == 1){
        mask_ = 2;
    } 

    // Default weights
    P_ = 1e-0*Eigen::VectorXd::Ones(nv_);
    Q_ = 1e-2*Eigen::VectorXd::Ones(nv_);
    R_ = 1e-2*Eigen::VectorXd::Ones(nc_);
    H_ = Eigen::MatrixXd::Zero(n_tot_, n_tot_);
    H_.bottomRightCorner(nv_, nv_) = P_.asDiagonal();
    H_.topLeftCorner(nv_, nv_) = Q_.asDiagonal();
    H_.block(nv_, nv_, nc_, nc_) = R_.asDiagonal();

    // QP solver
    qp_ = boost::make_shared<dense::QP<double>>(dense::QP<double>(n_tot_, neq_, nin_));

    if(baumgarte_gains_[0] > 1e-6){
        std::cout << "Error: the proportional gain of Baugmarte should be 0 !" << std::endl;
    }
    // std::cout << "Initialized force estimator." << std::endl;
}

TorqueEstimator::~TorqueEstimator(){}

void TorqueEstimator::estimate(
                const boost::shared_ptr<TorqueEstimatorData>& data, 
                const Eigen::Ref<const Eigen::VectorXd>& q,
                const Eigen::Ref<const Eigen::VectorXd>& v,
                const Eigen::Ref<const Eigen::VectorXd>& a,
                const Eigen::Ref<const Eigen::VectorXd>& tau,
                const Eigen::Ref<const Eigen::VectorXd>& dtau_prior,
                const Eigen::Ref<const Eigen::VectorXd>& F_mes){
    Data* d = static_cast<Data*>(data.get());
    
    // Compute required dynamic quantities
    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
    // pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q, v, a);
    pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q, v);
    pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);
    d->h = d->pinocchio.nle;
    d->M = d->pinocchio.M;
    // Copy upper triangular part into lower triangular part to get symmetric inertia matrix 
    d->M.triangularView<Eigen::StrictlyLower>() = d->M.transpose().triangularView<Eigen::StrictlyLower>();

    if(nc_ == 1){
        d->alpha0 = pinocchio::getFrameAcceleration(pinocchio_, d->pinocchio, frameId_, ref_).toVector().segment(mask_, nc_);
        d->nu = pinocchio::getFrameVelocity(pinocchio_, d->pinocchio, frameId_, ref_).toVector().segment(mask_, nc_);
        pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, frameId_, ref_, d->J);
        d->J1 = d->J.row(mask_);
    } else {
        d->alpha0 = pinocchio::getFrameAcceleration(pinocchio_, d->pinocchio, frameId_, ref_).toVector().head(nc_);
        d->nu = pinocchio::getFrameVelocity(pinocchio_, d->pinocchio, frameId_, ref_).toVector().head(nc_);
        pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, frameId_, ref_, d->J);
        d->J1 = d->J.topRows(nc_);
    }
    d->alpha0 -= baumgarte_gains_[1] * d->nu;

    // Construct QP
    d->b.topRows(nv_) = d->h - tau; 
    d->b.bottomRows(nc_) = -d->alpha0;
    d->A.topLeftCorner(nv_, nv_) = - d->M;
    d->A.block(0, nv_, nv_, nc_) = d->J1.transpose();
    d->A.block(0, nv_+nc_, nv_, nv_) = -Eigen::MatrixXd::Identity(nv_, nv_);
    d->A.block(nv_, 0, nc_, nv_) = d->J1;

    d->g.head(nv_) = -Q_.cwiseProduct(a);
    d->g.segment(nv_, nc_) = -R_.cwiseProduct(F_mes);
    d->g.tail(nv_)= - P_.cwiseProduct(dtau_prior);
    
    qp_->init(H_, d->g, d->A, d->b, d->C, d->l, d->u);

    qp_->solve();
    // std::cout << "optimal x: " << qp_->results.x << std::endl;
    d->delta_tau = qp_->results.x.bottomRows(nv_);
}

pinocchio::Model& TorqueEstimator::get_pinocchio() const{
    return pinocchio_;
}

std::size_t TorqueEstimator::get_nv() const {
    return nv_;
}

std::size_t TorqueEstimator::get_nc() const {
    return nc_;
}

std::size_t TorqueEstimator::get_nin() const {
    return nin_;
}

std::size_t TorqueEstimator::get_neq() const {
    return neq_;
}

const Eigen::Vector2d& TorqueEstimator::get_baumgarte_gains() const {
    return baumgarte_gains_;
}

std::size_t TorqueEstimator::get_frameId() const {
    return frameId_;
}

pinocchio::ReferenceFrame TorqueEstimator::get_ref() const {
    return ref_;
}

std::size_t TorqueEstimator::get_n_tot() const {
    return n_tot_;
}

std::size_t TorqueEstimator::get_mask() const {
    return mask_;
}

const Eigen::VectorXd& TorqueEstimator::get_P() const {
    return P_;
}

const Eigen::VectorXd& TorqueEstimator::get_Q() const {
    return Q_;
}

const Eigen::VectorXd& TorqueEstimator::get_R() const {
    return R_;
}

const Eigen::MatrixXd& TorqueEstimator::get_H() const {
    return H_;
}

void TorqueEstimator::set_nc(const std::size_t nc) {
    nc_ = nc;
}

void TorqueEstimator::set_baumgarte_gains(const Eigen::Vector2d& gains) {
    baumgarte_gains_ = gains;
}

void TorqueEstimator::set_frameId(const std::size_t frameId) {
    frameId_ = frameId;
}

void TorqueEstimator::set_ref(const pinocchio::ReferenceFrame ref) {
    ref_ = ref;
}

void TorqueEstimator::set_P(const Eigen::VectorXd& inP) {
    P_ = inP;
    H_.bottomRightCorner(nv_, nv_) = P_.asDiagonal();
}

void TorqueEstimator::set_Q(const Eigen::VectorXd& inQ) {
    Q_ = inQ;
    H_.topLeftCorner(nv_, nv_) = Q_.asDiagonal();
}

void TorqueEstimator::set_R(const Eigen::VectorXd& inR) {
    R_ = inR;
    H_.block(nv_, nv_, nc_, nc_) = R_.asDiagonal(); 
}

void TorqueEstimator::set_mask(const std::size_t mask) {
    mask_ = mask;
}


boost::shared_ptr<TorqueEstimatorData> TorqueEstimator::createData() {
//   return boost::allocate_shared<TorqueEstimatorData>(Eigen::aligned_allocator<TorqueEstimatorData>(), this);
  return boost::make_shared<TorqueEstimatorData>(this);
}

}  // namespace mim
}  // namespace estimator
