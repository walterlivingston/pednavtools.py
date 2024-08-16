import numpy as np
from numpy.linalg import inv
import quaternions as q

from ..imu import IMU, Noise
from ..pdr import detect_still
from .utils import initialPose, mag_madgwick
from ..utils import time_sync, skew

# (TODO) Implement Error-State Kalman Filter
def eskf(imu:IMU, noise:Noise):
    pass

# (TODO) Implement Multiplicative Extended Kalman Filter
def mekf(imu:IMU, noise:Noise):
    pass

def smekf(imu:IMU, noise:Noise):
    time, meas = time_sync(imu.time, imu.mag_time)
    N = time.size

    I = np.eye(3)
    O = np.zeros((3,3))
    g = np.transpose(np.array([0, 0, 9.81]))

    Rprime = np.diag(np.hstack((noise.sigma_a**2, noise.sigma_m**2)))
    Qc = np.diag(np.hstack((noise.sigma_g**2, noise.sigma_g**2)))
    P = Qc

    attBwrtRinQ = np.hstack((np.ones([N,1]), np.zeros([N,3])))
    # attBwrtRin321_0 = initialPose(imu)
    # attBwrtRinQ[0,...] = q.eul2q(attBwrtRin321_0)
    x = np.zeros((6))
    sigmas = np.ones((6,N))

    for i in range(1,N-1):
        # Process first IMU measurement before Magnetometer
        if not meas[0:i].all():
            # Process IMU Measurement
            if not meas[i-1]:
                idx = i - meas[0:i].sum()
                dt = time[idx] - time[idx-1]

                w = imu.gyr[...,idx]
                a = imu.acc[...,idx]

                # a priori orientation update
                attBwrtRinQ[i,...] = qKinematics(attBwrtRinQ[i-1,...], w, dt)
                toBfromR = q.q2DCM(attBwrtRinQ[i,...])

                # time update
                F = np.bmat([[-skew(w), -I],
                             [       O,  I]])
                Phi = np.eye(6) + F*dt
                Bw = np.bmat([[-I, O],
                              [ O, I]])
                P = Phi@P@Phi.T + Bw@Qc@Bw.T*dt

                # measurement update setup
                H = np.bmat([[skew(toBfromR@g), O]])
                R = Rprime[0:3,0:3]
                z_ = toBfromR@g - a
            # Process Magnetometer Measurement
            else:
                idx = meas[0:i].sum()
                dt = time[idx] - time[idx-1]
                attBwrtRinQ[i,...] = attBwrtRinQ[i-1,...]
                toBfromR = q.q2DCM(attBwrtRinQ[i,...])

                m = imu.mag[...,idx]
                q_b = mag_madgwick(attBwrtRinQ[i, ...], m)
                b = q_b[1:4]

                # measurement update setup
                H = np.bmat([[skew(toBfromR@b), O]])
                R = Rprime[3:6,3:6]
                z_ = toBfromR@b - m

            # Measurement Update
            L = P@H.T@inv(H@P@H.T + R)
            x = np.squeeze(L@z_)
            P = (np.eye(6) - L@H)@P

            q_a = np.array([1,x[0,0],x[0,1],x[0,2]]) / 2
            attBwrtRinQ[i,...] = q.qMult(attBwrtRinQ[i,...], q_a)
            attBwrtRinQ[i,...] = q.qNormalize(attBwrtRinQ[i,...])

            x = np.zeros((6))

    return np.squeeze(q.q2eul(attBwrtRinQ))

def qKinematics(qI:np.array, w:np.array, dt:float) -> np.array:
    q_w = np.array([0, w[0], w[1], w[2]])
    q0 = qI + 0.5*q.qMult(qI, q_w)*dt
    return q.qNormalize(q0)