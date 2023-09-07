
#include <pinocchio/parsers/urdf.hpp>
#include <ostream>
#include <Eigen/Dense>

#include "force_observer/mh_estimator.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_boost_estimator) {
    // Load robot model 
    // std::string URDF_PARAMS = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/iiwa_ft_sensor_shell.urdf";
    std::string URDF_PARAMS = "/home/skleff/.virtualenvs/croco_master_upstream/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/iiwa_ft_sensor_shell.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(URDF_PARAMS, model);

    // Params
    std::size_t nc = 1;
    pinocchio::FrameIndex frameId = model.getFrameId("contact");
    Eigen::Vector2d gains = Eigen::Vector2d::Zero();
    pinocchio::ReferenceFrame pinRef = pinocchio::LOCAL;
    std::size_t T = 10;
    mim::estimator::MHForceEstimator forceEstimator = mim::estimator::MHForceEstimator(T, model, nc, frameId, gains, pinRef);
    double TOL = 1e-6;

    // Check model default attributes
    BOOST_CHECK((unsigned int)forceEstimator.get_nv() - (unsigned int)model.nv  == 0.);
    BOOST_CHECK(forceEstimator.get_nc() == nc);
    BOOST_CHECK(forceEstimator.get_frameId() == frameId);
    BOOST_CHECK(forceEstimator.get_baumgarte_gains() == gains);
    BOOST_CHECK(forceEstimator.get_ref() == pinRef);

    BOOST_CHECK(forceEstimator.get_H().rows() == (unsigned int)forceEstimator.get_n_tot());
    BOOST_CHECK(forceEstimator.get_H().cols() == (unsigned int)forceEstimator.get_n_tot());
    BOOST_CHECK(forceEstimator.get_P().size() == (unsigned int)nc);
    BOOST_CHECK(forceEstimator.get_Q().size() == (unsigned int)model.nv);
    BOOST_CHECK(forceEstimator.get_R().size() == (unsigned int)nc);

    BOOST_CHECK( (forceEstimator.get_P() - 1e0*Eigen::VectorXd::Ones(nc)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_Q() - 1e-2*Eigen::VectorXd::Ones(model.nv)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_R() - 1e-2*Eigen::VectorXd::Ones(nc)).isZero(TOL) );
    BOOST_CHECK(forceEstimator.get_n_tot() == (unsigned int)nc + (unsigned int)T*((unsigned int)nc + (unsigned int)model.nv));
    BOOST_CHECK(forceEstimator.get_neq() == (unsigned int)T*((unsigned int)nc + (unsigned int)model.nv));
    BOOST_CHECK(forceEstimator.get_nin() == 0);

    // // Check construction of the QP
    // for(std::size_t t=0; t < T; t++){
    //     std::size_t ind = t * (model.nv + nc);
    //     std::cout << "t   = " << t << std::endl;
    //     std::cout << "ind = " << ind << std::endl;
    //     BOOST_CHECK( (forceEstimator.get_H().block(ind, ind, model.nv, model.nv) - Eigen::MatrixXd(forceEstimator.get_Q().asDiagonal())).isZero(TOL) );
    //     BOOST_CHECK( (forceEstimator.get_H().block(ind + model.nv, ind + model.nv, nc, nc) - Eigen::MatrixXd(forceEstimator.get_R().asDiagonal())).isZero(TOL) );
    //     std::cout << "H " << std::endl;
    //     std::cout << forceEstimator.get_H() << std::endl; //.block(ind + model.nv + nc, ind + model.nv + nc, nc, nc) << std::endl;
    //     std::cout << "R.diagonal() " << std::endl;
    //     std::cout << Eigen::MatrixXd(forceEstimator.get_R().asDiagonal()) << std::endl;
    //     BOOST_CHECK( (forceEstimator.get_H().block(ind + model.nv + nc, ind + model.nv + nc, nc, nc) - Eigen::MatrixXd(forceEstimator.get_R().asDiagonal())).isZero(TOL) );
    //     if(ind >= nc){
    //         BOOST_CHECK( (forceEstimator.get_H().block(ind + model.nv + nc, ind - nc, nc, nc) - Eigen::MatrixXd(forceEstimator.get_R().asDiagonal())).isZero(TOL));
    //     } 
    //     if((t+2) * (model.nv + nc)-nc < forceEstimator.get_n_tot()){
    //         BOOST_CHECK( (forceEstimator.get_H().block(ind + model.nv + nc, (t+2) * (model.nv + nc)-nc, nc, nc) - Eigen::MatrixXd(forceEstimator.get_R().asDiagonal()) ).isZero(TOL)); 
    //     }
    // }
    // BOOST_CHECK( (forceEstimator.get_H().bottomRightCorner(nc, nc) - Eigen::MatrixXd(forceEstimator.get_P().asDiagonal())).isZero(TOL));

    // Check setters
    gains = Eigen::Vector2d::Random();
    forceEstimator.set_baumgarte_gains(gains);
    BOOST_CHECK((forceEstimator.get_baumgarte_gains() - gains).isZero(TOL));

    pinRef = pinocchio::LOCAL_WORLD_ALIGNED;
    forceEstimator.set_ref(pinRef);
    BOOST_CHECK(forceEstimator.get_ref() == pinRef);

    frameId = model.getFrameId("A6");
    forceEstimator.set_frameId(frameId);
    BOOST_CHECK(forceEstimator.get_frameId() == frameId);

    std::size_t mask = 1;
    forceEstimator.set_mask(mask);
    BOOST_CHECK(forceEstimator.get_mask() == mask);

    Eigen::VectorXd P = Eigen::VectorXd::Random(nc);
    forceEstimator.set_P(P);
    BOOST_CHECK( (forceEstimator.get_P() - P).isZero(TOL) );

    Eigen::VectorXd Q = Eigen::VectorXd::Random(model.nv);
    forceEstimator.set_Q(Q);
    BOOST_CHECK( (forceEstimator.get_Q() - Q).isZero(TOL) );

    Eigen::VectorXd R = Eigen::VectorXd::Random(nc);
    forceEstimator.set_R(R);
    BOOST_CHECK( (forceEstimator.get_R() - R).isZero(TOL) );

    // Test estimator data
    boost::shared_ptr<mim::estimator::MHForceEstimatorData> forceEstimatorData = forceEstimator.createData();
    BOOST_CHECK( forceEstimatorData->F.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->delta_f.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->J.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->J1.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->alpha0.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->nu.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->M.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->h.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->b.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->A.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->g.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->C.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->l.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->u.isZero(TOL) );

    std::size_t nq = forceEstimator.get_pinocchio().nq;
    std::size_t nv = forceEstimator.get_pinocchio().nv;

    Eigen::VectorXd q_list = Eigen::VectorXd::Zero(nq * T);   
    Eigen::VectorXd v_list = Eigen::VectorXd::Zero(nv * T);   
    Eigen::VectorXd a_list = Eigen::VectorXd::Zero(nv * T);   
    Eigen::VectorXd tau_list = Eigen::VectorXd::Zero(nv * T);   
    Eigen::VectorXd F_mes_list = Eigen::VectorXd::Zero(nc * T);   

    for(std::size_t t = 0; t<T; t++){
        q_list.segment(t * nq, nq) = Eigen::VectorXd::Random(nq); 
        v_list.segment(t * nv, nv) = Eigen::VectorXd::Random(nv); 
        a_list.segment(t * nv, nv) = Eigen::VectorXd::Random(nv); 
        tau_list.segment(t * nv, nv) = Eigen::VectorXd::Random(nv);
        F_mes_list.segment(t * nc, nc) = Eigen::VectorXd::Random(nc);
    }
    Eigen::VectorXd df_prior = Eigen::VectorXd::Random(nc); 

    forceEstimator.estimate(forceEstimatorData, q_list, v_list, a_list, tau_list, df_prior, F_mes_list);
}

BOOST_AUTO_TEST_SUITE_END()