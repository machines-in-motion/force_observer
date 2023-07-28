
#include <pinocchio/parsers/urdf.hpp>
#include <ostream>
#include <Eigen/Dense>

#include "force_observer/mh_estimator.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_boost_estimator) {
    // Load robot model 
    std::string URDF_PARAMS = "/home/skleff/.virtualenvs/croco_master_upstream/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/iiwa_ft_sensor_shell.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(URDF_PARAMS, model);

    // Params
    std::size_t nc = 1;
    pinocchio::FrameIndex frameId = model.getFrameId("contact");
    std::cout << frameId << std::endl;
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
    BOOST_CHECK( (forceEstimator.get_P() - 5e-1*Eigen::VectorXd::Ones(nc)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_Q() - 1e-2*Eigen::VectorXd::Ones(model.nv)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_R() - 1e-2*Eigen::VectorXd::Ones(nc)).isZero(TOL) );
    BOOST_CHECK(forceEstimator.get_n_tot() == (unsigned int)nc + (unsigned int)T*((unsigned int)nc + (unsigned int)model.nv));
    BOOST_CHECK(forceEstimator.get_neq() == (unsigned int)T*((unsigned int)nc + (unsigned int)model.nv));
    BOOST_CHECK(forceEstimator.get_nin() == 0);

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

    std::vector<Eigen::VectorXd> q_list;   
    std::vector<Eigen::VectorXd> v_list;   
    std::vector<Eigen::VectorXd> a_list;   
    std::vector<Eigen::VectorXd> tau_list; 
    std::vector<Eigen::VectorXd> F_mes_list; 
    q_list.resize(T);
    v_list.resize(T);
    a_list.resize(T);
    tau_list.resize(T);
    F_mes_list.resize(T);
    for(std::size_t t = 0; t<T; t++){
        q_list[t] = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nq); 
        v_list[t] = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv); 
        a_list[t] = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv); 
        tau_list[t] = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv);
        F_mes_list[t] = Eigen::VectorXd::Random(1);
    }
    Eigen::VectorXd df_prior = Eigen::VectorXd::Random(1); 

    forceEstimator.estimate(forceEstimatorData, q_list, v_list, a_list, tau_list, df_prior, F_mes_list);
}

BOOST_AUTO_TEST_SUITE_END()