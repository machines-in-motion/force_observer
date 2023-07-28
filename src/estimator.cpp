#include "force_observer/estimator.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>

using namespace proxsuite::proxqp;

namespace mim {
namespace estimator {


ForceEstimator::ForceEstimator(
      pinocchio::Model& pin_model,
      std::size_t nc,
      std::size_t nc_delta_f,
      const pinocchio::FrameIndex frameId,
      const Vector2d& baumgarte_gains, 
      const pinocchio::ReferenceFrame ref
    ) : pinocchio_(pin_model),
        nc_(nc),
        nc_delta_f_(nc_delta_f),
        frameId_(frameId),
        baumgarte_gains_(baumgarte_gains),
        ref_(ref)    
        {
    // Dimensions
    nv_ = pinocchio_.nv;
    n_tot_ = nv_ + nc_ + nc_delta_f_;
    neq_ = nv_ + nc_;
    nin_ = 0.;

    // Set default mask if 1D model
    if(nc_ == 1){
        mask_ = 2;
    } 

    if(nc_ != nc_delta_f_){
        std::cout << "Error: nc must be equal to nc_delta_f !" << std::endl;
    }

    // Default weights
    P_ = 1e-0*Eigen::VectorXd::Ones(nc_delta_f_);
    Q_ = 1e-2*Eigen::VectorXd::Ones(nv_);
    R_ = 1e-2*Eigen::VectorXd::Ones(nc_);
    H_ = Eigen::MatrixXd::Zero(n_tot_, n_tot_);
    H_.bottomRightCorner(nc_delta_f_, nc_delta_f_) = P_.asDiagonal();
    H_.topLeftCorner(nv_, nv_) = Q_.asDiagonal();
    H_.block(nv_, nv_, nc_, nc_) = R_.asDiagonal();

    // QP solver
    qp_ = boost::make_shared<dense::QP<double>>(dense::QP<double>(n_tot_, neq_, nin_));

    if(baumgarte_gains_[0] > 1e-6){
        std::cout << "Error: the proportional gain of Baugmarte should be 0 !" << std::endl;
    }
    // std::cout << "Initialized force estimator." << std::endl;
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
    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
    pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q, v, a);
    pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);
    d->h = d->pinocchio.nle;
    d->M = d->pinocchio.M;
    // Copy upper triangular part into lower triangular part to get symmetric inertia matrix 
    d->M.triangularView<Eigen::StrictlyLower>() = d->M.transpose().triangularView<Eigen::StrictlyLower>();

    if(nc_ == 1){
        d->alpha0 = pinocchio::getFrameAcceleration(pinocchio_, d->pinocchio, frameId_).toVector().segment(mask_, nc_);
        d->nu = pinocchio::getFrameVelocity(pinocchio_, d->pinocchio, frameId_).toVector().segment(mask_, nc_);
        pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, frameId_, ref_, d->J);
        d->J1 = d->J.row(mask_);
    } else {
        d->alpha0 = pinocchio::getFrameAcceleration(pinocchio_, d->pinocchio, frameId_).toVector().head(nc_);
        d->nu = pinocchio::getFrameVelocity(pinocchio_, d->pinocchio, frameId_).toVector().head(nc_);
        pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, frameId_, ref_, d->J);
        d->J1 = d->J.topRows(nc_);
    }
    d->alpha0 -= baumgarte_gains_[1] * d->nu;
    if(nc_delta_f_ == 3 && nc_ == 1){
        pinocchio::getFrameJacobian(pinocchio_, d->pinocchio, frameId_, ref_, d->J);
        d->J2 = d->J.topRows(nc_delta_f_);
    } else {
        d->J2 = d->J1;
    }
    // Construct QP
    d->b.topRows(nv_) = d->h - tau; 
    d->b.bottomRows(nc_) = -d->alpha0;
    d->A.topLeftCorner(nv_, nv_) = d->M;
    d->A.block(0, nv_, nv_, nc_) = d->J1.transpose();
    d->A.block(0, nv_+nc_, nv_, nc_delta_f_) = d->J2.transpose();
    d->A.block(nv_, 0, nc_, nv_) = d->J1;

    d->g.head(nv_) = -Q_.cwiseProduct(a);
    d->g.segment(nv_, nc_) = -R_.cwiseProduct(F_mes);
    d->g.tail(nc_delta_f_)= - P_.cwiseProduct(df_prior);
    
    qp_->init(H_, d->g, d->A, d->b, d->C, d->l, d->u);

    qp_->solve();
    // std::cout << "optimal x: " << qp_->results.x << std::endl;
    d->delta_f = qp_->results.x.bottomRows(nc_delta_f_);
}

pinocchio::Model& ForceEstimator::get_pinocchio() const{
    return pinocchio_;
}

std::size_t ForceEstimator::get_nv() const {
    return nv_;
}

std::size_t ForceEstimator::get_nc() const {
    return nc_;
}

std::size_t ForceEstimator::get_nc_delta_f() const {
    return nc_delta_f_;
}

std::size_t ForceEstimator::get_nin() const {
    return nin_;
}

std::size_t ForceEstimator::get_neq() const {
    return neq_;
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

std::size_t ForceEstimator::get_n_tot() const {
    return n_tot_;
}

std::size_t ForceEstimator::get_mask() const {
    return mask_;
}

const Eigen::VectorXd& ForceEstimator::get_P() const {
    return P_;
}

const Eigen::VectorXd& ForceEstimator::get_Q() const {
    return Q_;
}

const Eigen::VectorXd& ForceEstimator::get_R() const {
    return R_;
}

const Eigen::MatrixXd& ForceEstimator::get_H() const {
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

void ForceEstimator::set_P(const Eigen::VectorXd& inP) {
    P_ = inP;
    H_.bottomRightCorner(nc_delta_f_, nc_delta_f_) = P_.asDiagonal();
}

void ForceEstimator::set_Q(const Eigen::VectorXd& inQ) {
    Q_ = inQ;
    H_.topLeftCorner(nv_, nv_) = Q_.asDiagonal();
}

void ForceEstimator::set_R(const Eigen::VectorXd& inR) {
    R_ = inR;
    H_.block(nv_, nv_, nc_, nc_) = R_.asDiagonal(); 
}

void ForceEstimator::set_mask(const std::size_t mask) {
    mask_ = mask;
}


boost::shared_ptr<ForceEstimatorData> ForceEstimator::createData() {
//   return boost::allocate_shared<ForceEstimatorData>(Eigen::aligned_allocator<ForceEstimatorData>(), this);
  return boost::make_shared<ForceEstimatorData>(this);
}

}  // namespace mim
}  // namespace estimator
