
#include <pinocchio/parsers/urdf.hpp>
#include <ostream>
#include <Eigen/Dense>

// #include "yaml_utils/yaml_cpp_fwd.hpp"
#include "force_observer/estimator.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(test_boost_estimator) {
    // Load robot model 
    std::string URDF_PARAMS = "/home/skleff/.virtualenvs/croco_master_upstream/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/iiwa_ft_sensor_shell.urdf";
    pinocchio::Model model;
    pinocchio::urdf::buildModel(URDF_PARAMS, model);

    // Params
    std::size_t nc = 3;
    std::size_t nc_delta_f = 3;
    pinocchio::FrameIndex frameId = model.getFrameId("contact");
    std::cout << frameId << std::endl;
    Eigen::Vector2d gains = Eigen::Vector2d::Zero();
    pinocchio::ReferenceFrame pinRef = pinocchio::LOCAL;
    mim::estimator::ForceEstimator forceEstimator = mim::estimator::ForceEstimator(model, nc, nc_delta_f, frameId, gains, pinRef);
    double TOL = 1e-6;

    // Check model default attributes
    BOOST_CHECK((unsigned int)forceEstimator.get_nv() - (unsigned int)model.nv  == 0.);
    BOOST_CHECK(forceEstimator.get_nc() == nc);
    BOOST_CHECK(forceEstimator.get_nc_delta_f() == nc_delta_f);
    BOOST_CHECK(forceEstimator.get_frameId() == frameId);
    BOOST_CHECK(forceEstimator.get_baumgarte_gains() == gains);
    BOOST_CHECK(forceEstimator.get_ref() == pinRef);
    BOOST_CHECK( (forceEstimator.get_P() - 1e-0*Eigen::VectorXd::Ones(nc_delta_f)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_Q() - 1e-2*Eigen::VectorXd::Ones(model.nv)).isZero(TOL) );
    BOOST_CHECK( (forceEstimator.get_R() - 1e-2*Eigen::VectorXd::Ones(nc)).isZero(TOL) );
    BOOST_CHECK(forceEstimator.get_n_tot() == (unsigned int)nc + (unsigned int)nc_delta_f + (unsigned int)model.nv);
    BOOST_CHECK(forceEstimator.get_neq() == (unsigned int)nc + (unsigned int)model.nv);
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

    Eigen::VectorXd P = Eigen::VectorXd::Random(nc_delta_f);
    forceEstimator.set_P(P);
    BOOST_CHECK( (forceEstimator.get_P() - P).isZero(TOL) );

    Eigen::VectorXd Q = Eigen::VectorXd::Random(model.nv);
    forceEstimator.set_Q(Q);
    BOOST_CHECK( (forceEstimator.get_Q() - Q).isZero(TOL) );

    Eigen::VectorXd R = Eigen::VectorXd::Random(nc);
    forceEstimator.set_R(R);
    BOOST_CHECK( (forceEstimator.get_R() - R).isZero(TOL) );

    // Test estimator data
    boost::shared_ptr<mim::estimator::ForceEstimatorData> forceEstimatorData = forceEstimator.createData();
    BOOST_CHECK( forceEstimatorData->F.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->delta_f.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->J.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->J1.isZero(TOL) );
    BOOST_CHECK( forceEstimatorData->J2.isZero(TOL) );
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

    Eigen::VectorXd q = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nq);
    Eigen::VectorXd v = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv);
    Eigen::VectorXd a = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Random(forceEstimator.get_pinocchio().nv);
    Eigen::Vector3d df_prior = Eigen::Vector3d::Random(); 
    Eigen::Vector3d F_mes = Eigen::Vector3d::Random(); 

    forceEstimator.estimate(forceEstimatorData, q, v, a, tau, df_prior, F_mes);
    // // Test estimate function
    // // BOOST_CHECK(gepetto::example::add(-3, 1) == -2);
}

BOOST_AUTO_TEST_SUITE_END()