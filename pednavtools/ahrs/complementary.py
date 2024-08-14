import numpy as np
from ..imu import IMU
from .utils import mag_madgwick
import quaternions as q

def madgwick(imu:IMU, beta:float=0.1, zeta:float=0) -> np.array:
    N = len(imu.time)
    q_est = np.vstack((np.ones([1,N]), np.zeros([3,N])))
    for i in range(N-1):
        dt = imu.time[i+1] - imu.time[i]
        q_est_dot = q_est[...,i]
        q_w = np.array([0,imu.gyr[0,i],imu.gyr[1,i],imu.gyr[2,i]])
        q_w = q.qNormalize(q_w)

        if imu.mag.size != 0:
            F,J = madgwick_jacobian(q_est[...,i], imu.acc[...,i], imu.gyr[...,i], imu.mag[...,i])
        else:
            F,J = madgwick_jacobian(q_est[...,i], imu.acc[...,i], imu.gyr[...,i])

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

def madgwick_jacobian(q_est:np.array, acc:np.array, gyr:np.array, mag:np.array = []):
    q_w = np.array([0,gyr[0],gyr[1],gyr[2]])
    q_w = q.qNormalize(q_w)
    q_a = np.array([0,acc[0],acc[1],acc[2]])
    q_a = q.qNormalize(q_a)

    Fg = np.zeros([3,1])
    Jg = np.zeros([3,4])

    Fg[0] = 2*(q_est[1]*q_est[3] - q_est[0]*q_est[2]) - q_a[1]
    Fg[1] = 2*(q_est[0]*q_est[1] + q_est[2]*q_est[3]) - q_a[2]
    Fg[2] = 2*(0.5 - q_est[1]**2 - q_est[2]**2) - q_a[3]

    Jg[0,0] = -2*q_est[2]
    Jg[0,1] =  2*q_est[3]
    Jg[0,2] = -2*q_est[0]
    Jg[0,3] =  2*q_est[1]

    Jg[1,0] =  2*q_est[1]
    Jg[1,1] =  2*q_est[0]
    Jg[1,2] =  2*q_est[3]
    Jg[1,3] =  2*q_est[2]

    Jg[2,0] =  0
    Jg[2,1] = -4*q_est[1]
    Jg[2,2] = -4*q_est[2]
    Jg[2,3] =  0

    if mag.size != 0:
        q_m = np.array([0, mag[0], mag[1], mag[2]])
        q_m = q.qNormalize(q_m)
        Fb = np.zeros([3,1])
        Jb = np.zeros([3,4])

        b = mag_madgwick(q_est, mag)

        Fb[0] = 2*b[1]*(0.5 - q_est[2]**2 - q_est[3]**2) + 2*b[3]*(q_est[1]*q_est[3] - q_est[0]*q_est[2]) - q_m[1]
        Fb[1] = 2*b[1]*(q_est[1]*q_est[2] - q_est[0]*q_est[3]) + 2*b[3]*(q_est[0]*q_est[1] + q_est[2]*q_est[3]) - q_m[2]
        Fb[2] = 2*b[1]*(q_est[0]*q_est[2] + q_est[1]*q_est[3]) + 2*b[3]*(0.5 - q_est[1]**2 - q_est[2]**2) - q_m[3]

        Jb[0,0] = -2*b[3]*q_est[2]
        Jb[0,1] =  2*b[3]*q_est[3]
        Jb[0,2] = -4*b[1]*q_est[2] - 2*b[3]*q_est[0]
        Jb[0,3] = -4*b[1]*q_est[1] + 2*b[3]*q_est[1]

        Jb[1,0] = -2*b[1]*q_est[3] + 2*b[3]*q_est[1]
        Jb[1,1] =  2*b[1]*q_est[2] + 2*b[3]*q_est[0]
        Jb[1,2] =  2*b[1]*q_est[1] + 2*b[3]*q_est[3]
        Jb[1,3] = -2*b[1]*q_est[0] + 2*b[3]*q_est[2]

        Jb[2,0] =  2*b[1]*q_est[2]
        Jb[2,1] =  2*b[1]*q_est[3] - 4*b[3]*q_est[1]
        Jb[2,2] =  2*b[1]*q_est[0] - 4*b[3]*q_est[2]
        Jb[2,3] =  2*b[1]*q_est[1]

        F = np.concatenate((Fg, Fb))
        J = np.concatenate((Jg, Jb))
    else:
        F = Fg
        J = Jg

    return F,J