import pinocchio as pin
import example_robot_data as robex
import numpy as np
from numpy.linalg import norm
np.random.seed(10)

from demos.estimator import Estimator, EstimatorEquivalent, MHEstimator, Varying_DF_MHEstimator
from force_observer import ForceEstimator, MHForceEstimator

np.set_printoptions(precision=4, linewidth=180)


# Estimators params
gains = np.zeros(2)

# Unittest tolerance
TOL   = 1e-3

# Test cases
ROBOTS = ['talos_arm', 'kinova']
FRAMES = ['wrist_left_ft_tool_link', 'j2s6s200_joint_finger_tip_1']

ROBOTS = ['kinova', 'talos_arm']
FRAMES = ['j2s6s200_joint_finger_tip_1', 'wrist_left_ft_tool_link']

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
            v = np.random.rand(robot.model.nv) 
            a = np.random.rand(robot.model.nv) 
            tau = np.random.rand(robot.model.nv) 
            f = np.random.rand(nc)
            df = np.random.rand(nc)
            nq = robot.model.nq
            nv = robot.model.nv
            id_endeff = robot.model.getFrameId(contactFrameName)

            print("\nTEST CASE "+robot_name+"_"+contactFrameName+"_"+pinRefFrameStr+"_NC="+str(nc))





            ####################################
            # Estimator vs EstimatorEquivalent #
            ####################################
            # i.e. compare if it is equivalent to have Df in the cost or in the constraint
            print("  >> test_Estimator_vs_EstimatorEquivalent")
            # Create estimators
            force_estimator_py    = Estimator(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
            force_estimator_eq_py = EstimatorEquivalent(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
            # Make sure they have the same params
            assert((np.diag(force_estimator_eq_py.P) == np.diag(force_estimator_py.P)).all())
            assert((np.diag(force_estimator_eq_py.Q) == np.diag(force_estimator_py.Q)).all())
            assert((np.diag(force_estimator_eq_py.R) == np.diag(force_estimator_py.R)).all())
            # Test that they give the same results
            _, delta_f_py    = force_estimator_py.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            _, delta_f_eq_py = force_estimator_eq_py.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            assert(norm(delta_f_py - delta_f_eq_py) <= TOL)




            #####################################
            # Estimator C++ vs Estimator Python #
            #####################################
            print("  >> test_ForceEstimator_CPP_vs_Estimator_PYTHON")

            # Test that we have same result as C++
            force_estimator_cpp = ForceEstimator(robot.model, nc, nc, id_endeff, gains, pinRefFrame)
            # Assert default values
            assert(force_estimator_cpp.frame_id == force_estimator_py.contact_frame_id)
            assert(force_estimator_cpp.nc == force_estimator_py.nc)
            assert(force_estimator_cpp.nc_delta_f == force_estimator_py.nc_delta_f)
            if(nc==1): assert(force_estimator_cpp.mask == force_estimator_py.mask)
            assert(norm(force_estimator_cpp.H - force_estimator_py.H) <= TOL)
            assert(norm(force_estimator_cpp.baumgarte_gains - force_estimator_py.baumgarte_gains) <= TOL)
            # Assert estimation
            force_estimator_cpp_data = force_estimator_cpp.createData()
            force_estimator_cpp.estimate(force_estimator_cpp_data, q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            
            assert(norm(force_estimator_cpp.H - force_estimator_py.H) <= TOL)
            assert(norm(force_estimator_cpp_data.g - force_estimator_py.g) <= TOL)
            assert(norm(force_estimator_cpp_data.A - force_estimator_py.A) <= TOL)
            assert(norm(force_estimator_cpp_data.b - force_estimator_py.b) <= TOL)            
            assert(norm(force_estimator_cpp_data.delta_f - delta_f_py) <= TOL)





            ##########################################################################
            # MHEstimator (T=1) vs EstimatorEquivalent (formulation with Df in cost) #
            ##########################################################################
            print("  >> test_MHEstimator_T=1_vs_EstimatorEquivalent")
            # Create estimators
            force_estimator_eq_py = EstimatorEquivalent(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
            force_estimator_mh_py = MHEstimator(1, robot, nc, id_endeff, gains, pinRefFrameStr)
            # Make sure they have the same params
            force_estimator_mh_py.P = force_estimator_eq_py.P
            force_estimator_mh_py.Q = force_estimator_eq_py.Q
            force_estimator_mh_py.R = force_estimator_eq_py.R
            # Test that they give the same results
            _, delta_f_eq_py    = force_estimator_eq_py.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            _, delta_f_mh_py    = force_estimator_mh_py.estimate([q.copy()], [v.copy()], [a.copy()], [tau.copy()], df.copy(), [f.copy()])
            assert(norm(force_estimator_mh_py.g - force_estimator_eq_py.g) <= TOL)
            assert(norm(force_estimator_mh_py.b - force_estimator_eq_py.b) <= TOL)
            assert(norm(force_estimator_mh_py.A - force_estimator_eq_py.A) <= TOL)
            assert(norm(delta_f_eq_py - delta_f_mh_py) <= TOL)




            ##########################################################################
            # MHEstimator (T=10) with a constant trajectory vs EstimatorEquivalent             #
            ##########################################################################
            print("  >> test_MHEstimator_T=10_with_a_constant_trajectory_vs_EstimatorEquivalent")
            # Create estimators
            T = 10
            force_estimator_eq_py = EstimatorEquivalent(robot, nc, nc, id_endeff, gains, pinRefFrameStr)
            force_estimator_mh_py = MHEstimator(T, robot, nc, id_endeff, gains, pinRefFrameStr)
            # Make sure they have the same params
            force_estimator_mh_py.P = force_estimator_eq_py.P
            force_estimator_mh_py.Q = force_estimator_eq_py.Q
            # balance weigth of T = 10
            force_estimator_mh_py.R = force_estimator_eq_py.R / T
            force_estimator_mh_py.define_parameters()
            # Test that they give the same results
            _, delta_f_eq_py    = force_estimator_eq_py.estimate(q.copy(), v.copy(), a.copy(), tau.copy(), df.copy(), f.copy())
            _, delta_f_mh_py    = force_estimator_mh_py.estimate([q.copy()] * T, [v.copy()] * T, [a.copy()] * T, [tau.copy()] * T, df.copy(), [f.copy()] * T)
            assert(norm(delta_f_eq_py - delta_f_mh_py) <= TOL)


            ##############################################
            # MHForceEstimator C++ vs MHEstimator Python #
            ##############################################
            print("  >> test_MHForceEstimator_CPP_vs_MHEstimator_PYTHON")
            force_estimator_mh_py  = MHEstimator(T, robot, nc, id_endeff, gains, pinRefFrameStr)
            # 
            force_estimator_mh_cpp = MHForceEstimator(T, robot.model, nc, id_endeff, gains, pinRefFrame)
            # Assert default values
            assert(force_estimator_mh_cpp.frame_id == force_estimator_mh_py.contact_frame_id)
            assert(force_estimator_mh_cpp.nc == force_estimator_mh_py.nc)
            if(nc==1): assert(force_estimator_mh_cpp.mask == force_estimator_mh_py.mask)


            # Make sure they have the same params
            assert((np.diag(force_estimator_mh_py.P) == force_estimator_mh_cpp.P).all())
            assert((np.diag(force_estimator_mh_py.Q) == force_estimator_mh_cpp.Q).all())
            assert((np.diag(force_estimator_mh_py.R) == force_estimator_mh_cpp.R).all())

            # Match matrix
            assert(norm(force_estimator_mh_cpp.H - force_estimator_mh_py.H) <= TOL)
            assert(norm(force_estimator_mh_cpp.baumgarte_gains - force_estimator_mh_py.baumgarte_gains) <= TOL)
            # Assert estimation
            force_estimator_mh_cpp_data = force_estimator_mh_cpp.createData()
            q_list = [] 
            v_list = [] 
            a_list = [] 
            tau_list = [] 
            f_list = [] 
            for i in range(T):
                q_list += list(q)
                v_list += list(v)
                a_list += list(a)
                tau_list += list(tau)
                f_list += list(f)
            force_estimator_mh_cpp.estimate(force_estimator_mh_cpp_data, np.array(q_list), np.array(v_list), np.array(a_list), np.array(tau_list), df.copy(), np.array(f_list))
            _, delta_f_mh_py    = force_estimator_mh_py.estimate([q.copy()] * T, [v.copy()] * T, [a.copy()] * T, [tau.copy()] * T, df.copy(), [f.copy()] * T)

            assert(norm(force_estimator_mh_cpp_data.delta_f - delta_f_mh_py) <= TOL)










print("OK !")



