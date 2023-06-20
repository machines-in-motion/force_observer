#include "force_observer/estimator.hpp"

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
        {}
    // std::cout << "Initialize force estimator"
// }

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
    d->delta_f = df_prior;
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


boost::shared_ptr<ForceEstimatorData> ForceEstimator::createData() {
  return boost::allocate_shared<ForceEstimatorData>(Eigen::aligned_allocator<ForceEstimatorData>(), this);
}

}  // namespace mim
}  // namespace estimator
