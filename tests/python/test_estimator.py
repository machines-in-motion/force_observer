import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.random.seed(10)
from demos.estimator import Estimator, MHEstimator, Varying_DF_MHEstimator

np.set_printoptions(precision=4, linewidth=180)


# Estimators params
gains = np.zeros(2)
nc    = 1

# Unittest tolerance
TOL   = 1e-3

# Test cases
ROBOTS = ['talos_arm', 'kinova']
FRAMES = ['wrist_left_ft_tool_link', 'j2s6s200_joint_finger_tip_1']

FRAMES_REF_STR = ['LOCAL', 'LOCAL_WORLD_ALIGNED']
FRAMES_REF     = [pin.LOCAL, pin.LOCAL_WORLD_ALIGNED]

# Main testing loop
for k,frame in enumerate(FRAMES_REF):
    pinRefFrameStr = FRAMES_REF_STR[k]
    pinRefFrame    = frame

    for k,robot_name in enumerate(ROBOTS):
        # Robot name and corresponding operational frame
        robot = robex.load(robot_name)
        contactFrameName = FRAMES[k]
        # Initial conditions
        q = pin.randomConfiguration(robot.model) 
        # Test C++ vs python fails when high velocities : different alpha0 and nu python vs C++ , maybe numpy numerical precision is the issue?
        v = np.random.rand(robot.model.nv)*0.01 
        # Careful ! In python estimator, zero acc is passed to forward kinematics (not the case in C++ estimator)
        a = np.zeros(robot.model.nv) 
        tau = np.random.rand(robot.model.nv) 
        f = np.random.rand(nc)
        df = np.zeros(nc)
        nq = robot.model.nq
        nv = robot.model.nv
        id_endeff = robot.model.getFrameId(contactFrameName)


        ##################################################################
        # MHE (T=1) vs classical (MHE : new formulation with Df in cost) #
        ##################################################################
        print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_MHE(T=1)_vs_classical")
        # Create estimators
        force_estimator = Estimator(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
        T_MHE = 1
        force_estimator_mh = MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrameStr)
        # Make sure they have the same params
        force_estimator_mh.P = force_estimator.P
        force_estimator_mh.Q = force_estimator.Q
        force_estimator_mh.R = force_estimator.R
        force_estimator_mh.H = force_estimator.H
        # Test that they give the same results
        _, delta_f = force_estimator.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
        _, delta_f_mh = force_estimator_mh.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], [df.copy()], [f.copy()])
        assert(norm(delta_f - delta_f_mh) <= TOL)


        ##########################################
        # MHE (T=1) with varying Df vs Classical #
        ##########################################
        print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_MHE(T=1)_varying_Df_vs_classical")
        T_MHE = 1
        force_estimator_mh_varying = Varying_DF_MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrameStr)
        # Make sure they have the same params
        force_estimator_mh_varying.P = force_estimator.P
        force_estimator_mh_varying.Q = force_estimator.Q
        force_estimator_mh_varying.R = force_estimator.R
        force_estimator_mh_varying.H = force_estimator.H
        _, delta_f_mh_varying = force_estimator_mh_varying.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], [df.copy()], [f.copy()])
        assert(norm(delta_f - delta_f_mh_varying) <= TOL)


        #####################################
        # Classical C++ vs Classical Python #
        #####################################
        print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_classical_C++_vs_classical_python")
        # Test that we have same result as C++
        from force_observer import ForceEstimator
        force_estimator_cpp = ForceEstimator(robot.model, nc, nc, id_endeff, gains, pinRefFrame)
        # Assert default values
        assert(force_estimator_cpp.frame_id == force_estimator.contact_frame_id)
        assert(force_estimator_cpp.nc == force_estimator.nc)
        assert(force_estimator_cpp.nc_delta_f == force_estimator.nc_delta_f)
        assert(force_estimator_cpp.mask == force_estimator.mask)
        assert(norm(force_estimator_cpp.H - force_estimator.H) <= TOL)
        assert(norm(force_estimator_cpp.baumgarte_gains - force_estimator.baumgarte_gains) <= TOL)
        # Assert estimation
        force_estimator_data = force_estimator_cpp.createData()
        force_estimator_cpp.estimate(force_estimator_data, q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
        assert(norm(force_estimator_data.delta_f - delta_f) <= TOL)


        #########################
        # MHE C++ vs MHE Python #
        #########################

print("OK !")



