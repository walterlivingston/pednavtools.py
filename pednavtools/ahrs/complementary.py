import numpy as np
from ..imu import IMU
import quaternions as q

def madgwick(imu:IMU, beta:float=0.1, zeta:float=0) -> np.array:
    N = len(imu.time)
    q_est = np.vstack((np.ones([1,N]), np.zeros([3,N])))
    for i in range(N-1):
        dt = imu.time[i+1] - imu.time[i]
        q_est_dot = q_est[...,i]
        q_w = np.array([0,imu.gyr[0,i],imu.gyr[1,i],imu.gyr[2,i]])
        q_w = q.qNormalize(q_w)
        q_a = np.array([0,imu.acc[0,i],imu.acc[1,i],imu.acc[2,i]])
        q_a = q.qNormalize(q_a)

        Fg = np.zeros([3,1])
        Jg = np.zeros([3,4])
        grad = np.array([1,0,0,0])

        Fg[0] = 2*(q_est[1,i]*q_est[3,i] - q_est[0,i]*q_est[2,i]) - q_a[1]
        Fg[1] = 2*(q_est[0,i]*q_est[1,i] + q_est[2,i]*q_est[3,i]) - q_a[2]
        Fg[2] = 2*(0.5 - q_est[1,i]**2 - q_est[2,i]**2) - q_a[3]

        Jg[0,0] = -2*q_est[2,i]
        Jg[0,1] =  2*q_est[3,i]
        Jg[0,2] = -2*q_est[0,i]
        Jg[0,3] =  2*q_est[1,i]

        Jg[1,0] =  2*q_est[1,i]
        Jg[1,1] =  2*q_est[0,i]
        Jg[1,2] =  2*q_est[3,i]
        Jg[1,3] =  2*q_est[2,i]

        Jg[2,0] =  0
        Jg[2,1] = -4*q_est[1,i]
        Jg[2,2] = -4*q_est[2,i]
        Jg[2,3] =  0

        if imu.mag.size != 0:
            q_m = np.array([0, imu.mag[0,i], imu.mag[1,i], imu.mag[2,i]])
            q_m = q.qNormalize(q_m)
            Fb = np.zeros([3,1])
            Jb = np.zeros([3,4])

            # h = q_est*q_m*q_est.conjugate()
            h = q.qMult(q_est[...,i], q.qMult(q_m, q.qConj(q_est[...,i])))
            b = np.array([0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3]])

            Fb[0] = 2*b[1]*(0.5 - q_est[2,i]**2 - q_est[3,i]**2) + 2*b[3]*(q_est[1,i]*q_est[3,i] - q_est[0,i]*q_est[2,i]) - q_m[1]
            Fb[1] = 2*b[1]*(q_est[1,i]*q_est[2,i] - q_est[0,i]*q_est[3,i]) + 2*b[3]*(q_est[0,i]*q_est[1,i] + q_est[2,i]*q_est[3,i]) - q_m[2]
            Fb[2] = 2*b[1]*(q_est[0,i]*q_est[2,i] + q_est[1,i]*q_est[3,i]) + 2*b[3]*(0.5 - q_est[1,i]**2 - q_est[2,i]**2) - q_m[3]

            Jb[0,0] = -2*b[3]*q_est[2,i]
            Jb[0,1] =  2*b[3]*q_est[3,i]
            Jb[0,2] = -4*b[1]*q_est[2,i] - 2*b[3]*q_est[0,i]
            Jb[0,3] = -4*b[1]*q_est[1,i] + 2*b[3]*q_est[1,i]

            Jb[1,0] = -2*b[1]*q_est[3,i] + 2*b[3]*q_est[1,i]
            Jb[1,1] =  2*b[1]*q_est[2,i] + 2*b[3]*q_est[0,i]
            Jb[1,2] =  2*b[1]*q_est[1,i] + 2*b[3]*q_est[3,i]
            Jb[1,3] = -2*b[1]*q_est[0,i] + 2*b[3]*q_est[2,i]

            Jb[2,0] =  2*b[1]*q_est[2,i]
            Jb[2,1] =  2*b[1]*q_est[3,i] - 4*b[3]*q_est[1,i]
            Jb[2,2] =  2*b[1]*q_est[0,i] - 4*b[3]*q_est[2,i]
            Jb[2,3] =  2*b[1]*q_est[1,i]

            F = np.concatenate((Fg, Fb))
            J = np.concatenate((Jg, Jb))
        else:
            F = Fg
            J = Jg

        grad_vec = np.transpose(J)@F
        grad = np.array([grad_vec[0], grad_vec[1], grad_vec[2], grad_vec[3]])
        grad = q.qNormalize(grad).flatten()

        q_wc = 2*q.qMult(q_est[...,i],grad)
        q_wc = q_w - zeta*q_wc
        q_est_dot = 0.5*q.qMult(q_est[...,i],q_wc)

        q_est_dot = q_est_dot - beta*grad
        q_est[...,i+1] = q_est[...,i+1] + q_est_dot*dt
        q_est[...,i+1] = q.qNormalize(q_est[...,i+1])
    att = np.squeeze(q.q2eul(q_est))
    return att