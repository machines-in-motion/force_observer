#include "force_observer/mh_estimator.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>

using namespace proxsuite::proxqp;

namespace mim {
namespace estimator {


MHForceEstimator::MHForceEstimator(
      std::size_t T,
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
    if(T <= 0){
        std::cout << "Error: horizon T must be > 0" << std::endl;
    }
    T_ = T;
    nv_ = pinocchio_.nv;
    n_tot_ = T_ * (nv_ + nc_) + nc_;
    neq_ = T_ * (nv_ + nc_);
    nin_ = 0.;

    // Set default mask if 1D model
    if(nc_ == 1){
        mask_ = 2;
    } 

    // Default weights
    P_ = 1e0*Eigen::VectorXd::Ones(nc_);
    Q_ = 1e-2*Eigen::VectorXd::Ones(nv_);
    R_ = 1e-2*Eigen::VectorXd::Ones(nc_);
    H_ = Eigen::MatrixXd::Zero(n_tot_, n_tot_);

    
    for(std::size_t t=0; t < T_; t++){
        // diagonal terms
        std::size_t ind = t * (nv_ + nc_);
        H_.block(ind, ind, nv_, nv_)  = Q_.asDiagonal();
        
        H_.block(ind + nv_, ind + nv_, nc_, nc_) = R_.asDiagonal(); 
        // last row
        H_.bottomRows(nc_).block(0, ind + nv_, nc_, nc_) = (- R_).asDiagonal(); 
        // last col
        H_.rightCols(nc_).block(ind + nv_, 0, nc_, nc_) = (- R_).asDiagonal();
    }
    std::cout <<  "H" <<  H_.bottomRightCorner(nc_, nc_) << std::endl;
    H_.bottomRightCorner(nc_, nc_) = P_.asDiagonal();
    std::cout <<  "H" <<  H_.bottomRightCorner(nc_, nc_) << std::endl;
    H_.bottomRightCorner(nc_, nc_) += (T_ * R_).asDiagonal();
    std::cout <<  "H" <<  H_.bottomRightCorner(nc_, nc_) << std::endl;

    // QP solver
    qp_ = boost::make_shared<dense::QP<double>>(dense::QP<double>(n_tot_, neq_, nin_));

    if(baumgarte_gains_[0] > 1e-6){
        std::cout << "Error: the proportional gain of Baugmarte should be 0 !" << std::endl;
    }
    // std::cout << "Initialized force estimator." << std::endl;
}

MHForceEstimator::~MHForceEstimator(){}

void MHForceEstimator::estimate(
                const boost::shared_ptr<MHForceEstimatorData>& data, 
                const Eigen::Ref<const Eigen::VectorXd>& q_list,
                const Eigen::Ref<const Eigen::VectorXd>& v_list,
                const Eigen::Ref<const Eigen::VectorXd>& a_list,
                const Eigen::Ref<const Eigen::VectorXd>& tau_list,
                const Eigen::Ref<const Eigen::VectorXd>& df_prior,
                const Eigen::Ref<const Eigen::VectorXd>& F_mes_list){
    Data* d = static_cast<Data*>(data.get());
    
    for(std::size_t t=0; t < T_; t++){

        // Compute required dynamic quantities
        pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q_list.segment(t * nv_, nv_), v_list.segment(t * nv_, nv_));
        pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q_list.segment(t * nv_, nv_), v_list.segment(t * nv_, nv_), a_list.segment(t * nv_, nv_));
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

        std::size_t ind =  t * (nv_ + nc_);

        // Construct QP
        d->b.segment(ind, nv_) = d->h - tau_list.segment(t * nv_, nv_); 
        d->b.segment(ind+nv_, nc_) = -d->alpha0;
        d->A.block(ind, ind, nv_, nv_) = -d->M;
        d->A.block(ind, ind+nv_, nv_, nc_) = d->J1.transpose();
        d->A.block(ind+nv_, ind, nc_, nv_) = d->J1;
        d->g.segment(ind, nv_) = -Q_.cwiseProduct(a_list.segment(t * nv_, nv_));
        d->g.segment(ind+nv_, nc_) = -R_.cwiseProduct(F_mes_list.segment(t * nv_, nv_));
        d->g.segment(ind+nv_+nc_, nc_) += -R_.cwiseProduct(F_mes_list.segment(t * nv_, nv_));
    }

    d->g.tail(nc_)= -P_.cwiseProduct(df_prior);
    
    qp_->init(H_, d->g, d->A, d->b, d->C, d->l, d->u); //std::nullopt, std::nullopt, std::nullopt); nullopt with C++17 only !

    qp_->solve();
    // std::cout << "optimal x: " << qp_->results.x << std::endl;
    d->delta_f = qp_->results.x.bottomRows(nc_);
}

std::size_t MHForceEstimator::get_T() const {
    return T_;
}

pinocchio::Model& MHForceEstimator::get_pinocchio() const{
    return pinocchio_;
}

std::size_t MHForceEstimator::get_nv() const {
    return nv_;
}

std::size_t MHForceEstimator::get_nc() const {
    return nc_;
}

std::size_t MHForceEstimator::get_nin() const {
    return nin_;
}

std::size_t MHForceEstimator::get_neq() const {
    return neq_;
}

const Eigen::Vector2d& MHForceEstimator::get_baumgarte_gains() const {
    return baumgarte_gains_;
}

std::size_t MHForceEstimator::get_frameId() const {
    return frameId_;
}

pinocchio::ReferenceFrame MHForceEstimator::get_ref() const {
    return ref_;
}

std::size_t MHForceEstimator::get_n_tot() const {
    return n_tot_;
}

std::size_t MHForceEstimator::get_mask() const {
    return mask_;
}

const Eigen::VectorXd& MHForceEstimator::get_P() const {
    return P_;
}

const Eigen::VectorXd& MHForceEstimator::get_Q() const {
    return Q_;
}

const Eigen::VectorXd& MHForceEstimator::get_R() const {
    return R_;
}

const Eigen::MatrixXd& MHForceEstimator::get_H() const {
    return H_;
}

void MHForceEstimator::set_nc(const std::size_t nc) {
    nc_ = nc;
}

void MHForceEstimator::set_baumgarte_gains(const Eigen::Vector2d& gains) {
    baumgarte_gains_ = gains;
}

void MHForceEstimator::set_frameId(const std::size_t frameId) {
    frameId_ = frameId;
}

void MHForceEstimator::set_ref(const pinocchio::ReferenceFrame ref) {
    ref_ = ref;
}

void MHForceEstimator::set_P(const Eigen::VectorXd& inP) {
    P_ = inP;
    H_.bottomRightCorner(nc_, nc_) = P_.asDiagonal();
    H_.bottomRightCorner(nc_, nc_) += R_.asDiagonal();
}

void MHForceEstimator::set_Q(const Eigen::VectorXd& inQ) {
    Q_ = inQ;
    for(std::size_t t=0; t < T_; t++){
        std::size_t ind = t * (nv_ + nc_);
        if(ind < T_-1){
            H_.block(ind, ind, nv_, nv_)  = Q_.asDiagonal();
        }
    }
}

void MHForceEstimator::set_R(const Eigen::VectorXd& inR) {
    R_ = inR;
    for(std::size_t t=0; t < T_; t++){
        // diagonal terms
        std::size_t ind = t * (nv_ + nc_);
        if(ind < T_-1){
            H_.block(ind, ind, nv_, nv_)  = Q_.asDiagonal();
        }
        H_.block(ind + nv_, ind + nv_, nc_, nc_) = R_.asDiagonal(); 
        // last row
        H_.bottomRows(nc_).block(0, ind + nv_, nc_, nc_) = R_.asDiagonal(); 
        // last col
        H_.rightCols(nc_).block(ind + nv_, 0, nc_, nc_) = R_.asDiagonal();
    }
    H_.bottomRightCorner(nc_, nc_) = P_.asDiagonal();
    H_.bottomRightCorner(nc_, nc_) += R_.asDiagonal();
}

void MHForceEstimator::set_mask(const std::size_t mask) {
    mask_ = mask;
}


boost::shared_ptr<MHForceEstimatorData> MHForceEstimator::createData() {
  return boost::make_shared<MHForceEstimatorData>(this);
}

}  // namespace mim
}  // namespace estimator
