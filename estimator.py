'''
Force Estimator
'''

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import pin_utils, mpc_utils

from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import pinocchio as pin
import proxsuite
import time


class Estimator():
    '''
    Computes the forward dynamics under rigid contact model 6D + force estimate
    '''
    def __init__(self, baumgarte_gains):
        self.R = 1e-2 * np.eye(6)
        self.Q = 1e-2 * np.eye(7)
        self.P = 1e-0 * np.eye(6)

        n = 19
        n_eq = 7 + 6
        n_in = 0

        self.qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)


        self.H = np.zeros((19, 19))
        self.H[:7, :7]  = self.Q
        self.H[7:13, 7:13]  = self.R 
        self.H[13:, 13:]  = self.P 


        self.n = n

        self.C = None
        self.u = None
        self.l = None

        self.baumgarte_gains = baumgarte_gains

        assert self.baumgarte_gains[0] == 0

    def estimate(self, pin_robot, q, v, a, tau, df_prior, F_mes):

        M = pin_robot.mass(q)
        contact_frame_id = pin_robot.model.getFrameId("contact")
        J = pin_robot.computeFrameJacobian(q, contact_frame_id)
        h = pin_robot.nle(q, v)


        pin.forwardKinematics(pin_robot.model, pin_robot.data, q, v, np.zeros(7))
        pin.updateFramePlacements(pin_robot.model, pin_robot.data)
        alpha0 = pin.getFrameAcceleration(pin_robot.model, pin_robot.data, contact_frame_id).vector
        v = pin.getFrameVelocity(pin_robot.model, pin_robot.data, contact_frame_id)

        alpha0 -= self.baumgarte_gains[1] * v.vector

        b = np.concatenate([h - tau, -alpha0], axis=0)

        A1 = np.concatenate([-M, J.T, J.T], axis=1)
        A2 = np.concatenate([J, np.zeros((6, 12))], axis=1)
        # import pdb; pdb.set_trace()
        A = np.concatenate([A1, A2], axis=0)


        # n_eq = A.shape[0]

        g = np.zeros(19)
        # import pdb; pdb.set_trace()
        g[:7] = - self.Q @ a
        g[7:13] = - self.R @ F_mes
        g[13:] = - self.P @ df_prior

        # solve it
        
        self.qp.init(self.H, g, A, b, self.C, self.l, self.u)



        t1 = time.time()
        self.qp.solve()
        # print("time ", time.time() - t1)
        # print("iter ", self.qp.results.info.iter)
        # print("primal ", self.qp.results.info.pri_res)
        # print("dual ", self.qp.results.info.dua_res)



        # self.qp.settings.initial_guess = (
        #     proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        # )

        # # # print an optimal solution
        # # print("optimal x: {}".format(qp.results.x))
        # # print("optimal y: {}".format(qp.results.y))
        # # print("optimal z: {}".format(qp.results.z))


        # t1 = time.time()

        # self.qp.solve()
        # print("time ", time.time() - t1)
        # print("iter ", self.qp.results.info.iter)
        # print("primal ", self.qp.results.info.pri_res)
        # print("dual ", self.qp.results.info.dua_res)
        # print(1/0)

        return self.qp.results.x[7:13], self.qp.results.x[13:]




# class Estimator():
#     '''
#     Computes the forward dynamics under rigid contact model 6D + force estimate
#     '''
#     def __init__(self):
#         self.R = 1e-2 * np.eye(6)
#         self.P = 1e-0 * np.eye(6)

#         n = 12
#         n_eq = 7
#         n_in = 0

#         self.qp = proxsuite.proxqp.dense.QP(n, n_eq, n_in)


#         self.H = np.zeros((12, 12))
#         self.H[:6, :6]  = self.R 
#         self.H[6:, 6:]  = self.P 


#         self.n = 12

#         self.C = None
#         self.u = None
#         self.l = None

#     def estimate(self, pin_robot, q, v, a, tau, df_prior, F_mes):

#         M = pin_robot.mass(q)
#         contact_frame_id = pin_robot.model.getFrameId("contact")
#         J = pin_robot.computeFrameJacobian(q, contact_frame_id)
#         h = pin_robot.nle(q, v)


#         b = M @ a + h - tau

#         A = np.concatenate([J.T, J.T], axis=1)
        
#         print(np.linalg.matrix_rank(A))

#         # n_eq = A.shape[0]

#         g = np.zeros(12)
#         # import pdb; pdb.set_trace()
#         g[:6] = - self.R @ F_mes
#         g[6:] = - self.P @ df_prior

#         # solve it
        
#         self.qp.init(self.H, g, A, b, self.C, self.l, self.u)



#         t1 = time.time()
#         self.qp.solve()
#         print("time ", time.time() - t1)
#         print("iter ", self.qp.results.info.iter)
#         print("primal ", self.qp.results.info.pri_res)
#         print("dual ", self.qp.results.info.dua_res)



#         # self.qp.settings.initial_guess = (
#         #     proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
#         # )

#         # # # print an optimal solution
#         # # print("optimal x: {}".format(qp.results.x))
#         # # print("optimal y: {}".format(qp.results.y))
#         # # print("optimal z: {}".format(qp.results.z))


#         # t1 = time.time()

#         # self.qp.solve()
#         # print("time ", time.time() - t1)
#         # print("iter ", self.qp.results.info.iter)
#         # print("primal ", self.qp.results.info.pri_res)
#         # print("dual ", self.qp.results.info.dua_res)
#         # print(1/0)

#         return self.qp.results.x[:6], self.qp.results.x[6:]

