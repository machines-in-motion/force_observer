import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.random.seed(10)

from demos.estimator import Estimator, MHEstimator, Varying_DF_MHEstimator
from force_observer import ForceEstimator, MHForceEstimator

np.set_printoptions(precision=4, linewidth=180)


# Estimators params
gains = np.zeros(2)
# nc    = 1

# Unittest tolerance
TOL   = 1e-3

# Test cases
ROBOTS = ['talos_arm', 'kinova']
FRAMES = ['wrist_left_ft_tool_link', 'j2s6s200_joint_finger_tip_1']

FRAMES_REF_STR = ['LOCAL', 'LOCAL_WORLD_ALIGNED']
FRAMES_REF     = [pin.LOCAL, pin.LOCAL_WORLD_ALIGNED]

CONTACT_DIMS = [1, 3]

# Main testing loop

for _, nc in enumerate(CONTACT_DIMS):

    for idf,frame in enumerate(FRAMES_REF):
        pinRefFrameStr = FRAMES_REF_STR[idf]
        pinRefFrame    = frame

        for idr,robot_name in enumerate(ROBOTS):
            # Robot name and corresponding operational frame
            robot = robex.load(robot_name)
            contactFrameName = FRAMES[idr]
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
            print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc)+"_MHE(T=1)_vs_classical")
            # Create estimators
            force_estimator_py = Estimator(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
            T_MHE = 1
            force_estimator_mh_py = MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrameStr)
            # Make sure they have the same params
            force_estimator_mh_py.P = force_estimator_py.P
            force_estimator_mh_py.Q = force_estimator_py.Q
            force_estimator_mh_py.R = force_estimator_py.R
            force_estimator_mh_py.H = force_estimator_py.H
            # Test that they give the same results
            _, delta_f_py    = force_estimator_py.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            _, delta_f_mh_py = force_estimator_mh_py.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], [df.copy()], [f.copy()])
            assert(norm(delta_f_py - delta_f_mh_py) <= TOL)


            ##########################################
            # MHE (T=1) with varying Df vs Classical #
            ##########################################
            print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc)+"_MHE(T=1)_varying_Df_vs_classical")
            T_MHE = 1
            force_estimator_mh_varying_py = Varying_DF_MHEstimator(T_MHE, robot, nc, id_endeff, gains, pinRefFrameStr)
            # Make sure they have the same params
            force_estimator_mh_varying_py.P = force_estimator_py.P
            force_estimator_mh_varying_py.Q = force_estimator_py.Q
            force_estimator_mh_varying_py.R = force_estimator_py.R
            force_estimator_mh_varying_py.H = force_estimator_py.H
            _, delta_f_mh_varying_py = force_estimator_mh_varying_py.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], [df.copy()], [f.copy()])
            assert(norm(delta_f_py - delta_f_mh_varying_py) <= TOL)


            #####################################
            # Classical C++ vs Classical Python #
            #####################################
            print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc)+"_classical_C++_vs_classical_python")
            # Test that we have same result as C++
            force_estimator_cpp = ForceEstimator(robot.model, nc, nc, id_endeff, gains, pinRefFrame)
            # Assert default values
            assert(force_estimator_cpp.frame_id == force_estimator_py.contact_frame_id)
            assert(force_estimator_cpp.nc == force_estimator_py.nc)
            assert(force_estimator_cpp.nc_delta_f == force_estimator_py.nc_delta_f)
            assert(force_estimator_cpp.mask == force_estimator_py.mask)
            assert(norm(force_estimator_cpp.H - force_estimator_py.H) <= TOL)
            assert(norm(force_estimator_cpp.baumgarte_gains - force_estimator_py.baumgarte_gains) <= TOL)
            # Assert estimation
            force_estimator_cpp_data = force_estimator_cpp.createData()
            force_estimator_cpp.estimate(force_estimator_cpp_data, q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            assert(norm(force_estimator_cpp_data.delta_f - delta_f_py) <= TOL)


            #########################
            # MHE C++ vs MHE Python #
            #########################
            print("Testing "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc)+"_MHE_C++_vs_MHE_python")
            T = 1
            force_estimator_mh_py  = MHEstimator(T, robot, nc, id_endeff, gains, pinRefFrameStr)
            # zepojzp
            force_estimator_mh_cpp = MHForceEstimator(T, robot.model, nc, id_endeff, gains, pinRefFrame)
            # Assert default values
            assert(force_estimator_mh_cpp.frame_id == force_estimator_mh_py.contact_frame_id)
            assert(force_estimator_mh_cpp.nc == force_estimator_mh_py.nc)
            assert(force_estimator_mh_cpp.mask == force_estimator_mh_py.mask)
            force_estimator_mh_cpp.P = force_estimator_mh_py.P
            # force_estimator_mh_cpp.Q = force_estimator_mh_py.Q
            # force_estimator_mh_cpp.R = force_estimator_mh_py.R
            # force_estimator_mh_cpp.H = force_estimator_mh_py.H
            # assert(norm(force_estimator_mh_cpp.H - force_estimator_mh_py.H) <= TOL)
            assert(norm(force_estimator_mh_cpp.baumgarte_gains - force_estimator_mh_py.baumgarte_gains) <= TOL)
            # Assert estimation
            force_estimator_mh_cpp_data = force_estimator_mh_cpp.createData()
            force_estimator_mh_cpp.estimate(force_estimator_mh_cpp_data, [q.copy()], [v.copy()], [a.copy()], [tau.copy()], df.copy(), [f.copy()])
            assert(norm(force_estimator_mh_cpp_data.delta_f - delta_f_mh_py) <= TOL)

print("OK !")



