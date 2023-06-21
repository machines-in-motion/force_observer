#ifndef __force_observer_estimator__
#define __force_observer_estimator__

#include <pinocchio/fwd.hpp>
// Include pinocchio first
#include <Eigen/Dense>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <string>
#include <vector>

#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include <proxsuite/helpers/optional.hpp> // for c++14
#include <proxsuite/proxqp/dense/dense.hpp>

using namespace proxsuite::proxqp;
using proxsuite::nullopt; // c++17 simply use std::nullopt

namespace mim {
namespace estimator {

struct ForceEstimatorData {

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename Eigen::VectorXd VectorXd;
  typedef typename Eigen::MatrixXd MatrixXd;

  template <class Model>
  explicit ForceEstimatorData(Model* const model)
      : pinocchio(pinocchio::Data(model->get_pinocchio())),
        F(model->get_nc()),
        delta_f(model->get_nc_delta_f()),
        J(6, model->get_pinocchio().nv),
        J1(model->get_nc(), model->get_pinocchio().nv),
        J2(model->get_nc_delta_f(), model->get_pinocchio().nv),
        alpha0(model->get_nc()),
        nu(model->get_nc()),
        M(model->get_pinocchio().nv, model->get_pinocchio().nv),
        h(model->get_pinocchio().nv),
        b(model->get_pinocchio().nv + model->get_nc()),
        A(model->get_pinocchio().nv + model->get_nc(), 
          model->get_pinocchio().nv + model->get_nc() + model->get_nc_delta_f()),
        g(model->get_n_tot()),
        C(model->get_nin(), model->get_n_tot()),
        l(model->get_nin()),
        u(model->get_nin()) {
      F.setZero();
      delta_f.setZero();
      J.setZero();
      J1.setZero();
      J2.setZero();
      alpha0.setZero();
      nu.setZero();
      M.setZero();
      h.setZero();
      b.setZero();
      A.setZero();
      g.setZero();
      C.setZero();
      l.setZero();
      u.setZero();
    }

  virtual ~ForceEstimatorData() {}
  
  pinocchio::Data pinocchio; //!< Pinocchio data

  VectorXd F;       //!< Force
  VectorXd delta_f; //!< Force offset estimate
  MatrixXd J;       //!< Full Jacobian
  MatrixXd J1;      //!< Jacobian for predicted force
  MatrixXd J2;      //!< Jacobien for force offset
  VectorXd alpha0;  //!< Contact frmae acceleration drift
  VectorXd nu;      //!< Contact frame velocity
  MatrixXd M;       //!< Generalized inertia matrix
  VectorXd h;       //!< Contact frame velocity

  VectorXd b;      //!< QP param
  MatrixXd A;     //!< QP param
  VectorXd g;      //!< QP param
  MatrixXd C;      //!< QP param
  VectorXd l;      //!< QP param
  VectorXd u;      //!< QP param
};



/**
 * @brief Force estimator
 */

class ForceEstimator{
  
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef typename Eigen::Vector2d Vector2d;
  typedef typename Eigen::VectorXd VectorXd;
  typedef typename Eigen::MatrixXd MatrixXd;
  typedef ForceEstimatorData Data;


  /**
   * @brief Instantiates the force estimator
   *
   * @param[in] pin_model         Pinocchio model
   * @param[in] nc                Dimension of the contact model
   * @param[in] nc_delta_f        Dimension of the force offset
   * @param[in] frameId           Contact frame id
   * @param[in] baumgarte_gains   Baumgarte gains in contact model
   * @param[in] ref               Pinocchio reference frame of the contact model
   */
  ForceEstimator(
      boost::shared_ptr<pinocchio::Model> pin_model,
      std::size_t nc,
      std::size_t nc_delta_f,
      const pinocchio::FrameIndex frameId,
      const Vector2d& baumgarte_gains, 
      const pinocchio::ReferenceFrame ref = pinocchio::LOCAL);

  ~ForceEstimator();

  /**
   * @brief Computes the force estimate from measurements and prior
   *
   * @param[in] data     Force estimator data
   * @param[in] q        Joint positions \f$\mathbf{x}\in\mathbb{R}^{nq}\f$
   * @param[in] v        Joint velocities Force point \f$\mathbf{f}\in\mathbb{R}^{nv}\f$
   * @param[in] a        Joint accelerations \f$\mathbf{u}\in\mathbb{R}^{nv}\f$
   * @param[in] tau      Joint torques input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   * @param[in] df_prior Prior \f$\mathbf{u}\in\mathbb{R}^{nc}\f$
   * @param[in] F_mes    Measured force \f$\mathbf{u}\in\mathbb{R}^{nc_delta_f}\f$
   */
  void estimate(const boost::shared_ptr<ForceEstimatorData>& data, 
                const Eigen::Ref<const VectorXd>& q,
                const Eigen::Ref<const VectorXd>& v,
                const Eigen::Ref<const VectorXd>& a,
                const Eigen::Ref<const VectorXd>& tau,
                const Eigen::Ref<const VectorXd>& df_prior,
                const Eigen::Ref<const VectorXd>& F_mes);

    /**
   * @brief Create the force estimator data
   *
   * @return Force estimator data
   */
  virtual boost::shared_ptr<ForceEstimatorData> createData();

  // getters 
  pinocchio::Model& get_pinocchio() const;
  std::size_t get_nv() const;
  std::size_t get_nc() const;
  std::size_t get_nc_delta_f() const;
  const Vector2d& get_baumgarte_gains() const;
  std::size_t get_frameId() const;
  pinocchio::ReferenceFrame get_ref() const;

  std::size_t get_n_tot() const;
  std::size_t get_neq() const;
  std::size_t get_nin() const;
  const VectorXd& get_P() const;
  const VectorXd& get_Q() const;
  const VectorXd& get_R() const;
  const MatrixXd& get_H() const;

  // setters
  void set_nv(const std::size_t);
  void set_nc(const std::size_t);
  void set_nc_delta_f(const std::size_t);
  void set_frameId(const std::size_t ref);
  void set_baumgarte_gains(const Vector2d&);
  void set_ref(const pinocchio::ReferenceFrame ref);

  void set_P(const VectorXd& p);
  void set_Q(const VectorXd& q);
  void set_R(const VectorXd& r);

 protected:
    pinocchio::Model& pinocchio_;                   //!< Pinocchio model
    std::size_t nc_;                                //!< Dimension of measured force
    std::size_t nc_delta_f_;                        //!< Dimension of estimated force offset
    std::size_t frameId_;                           //!< Contact frame id
    Vector2d baumgarte_gains_;                      //!< Baumgarte gains
    pinocchio::ReferenceFrame ref_;                 //!< Pinocchio reference frame    

    std::size_t n_tot_;                             //!< Total QP dimension
    std::size_t nv_;                                //!< Joint vel dimension 
    std::size_t neq_;                               //!< Number of equality constraints in the QP 
    std::size_t nin_;                               //!< Number of inequality constraints in the QP
    std::size_t mask_;                              //!< Mask of the 1D constraint (only for nc=1)

    VectorXd P_;                                    //!< Force offset weight         
    VectorXd Q_;                                    //!< Joint acceleration weight         
    VectorXd R_;                                    //!< Force weight
    MatrixXd H_;                                    //!< QP param         

    boost::shared_ptr<dense::QP<double>> qp_;                          //!< QP solver
};


}  // namespace mim
}  // namespace estimator


#endif  // __force_observer_estimator__