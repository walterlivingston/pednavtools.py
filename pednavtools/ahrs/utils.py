import numpy as np
import quaternions as q

from ..imu import IMU
from ..pdr import detect_still

def mag_madgwick(q_est:np.array, mag:np.array):
    q_m = np.array([0, mag[0], mag[1], mag[2]])
    q_m = q.qNormalize(q_m)
    h = q.qMult(q_est, q.qMult(q_m, q.qConj(q_est)))
    b = np.array([0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3]])
    return b

def initialPose(imu:IMU):
    cal_range = detect_still(imu)

    acc_ = np.mean(imu.acc[...,cal_range], axis=1)
    mag_ = np.mean(imu.mag[...,cal_range], axis=1)

    roll = np.atan(-acc_[0]/np.sqrt(acc_[1]**2 + acc_[2]**2))
    pitch = np.atan2(-acc_[1], -acc_[0])
    yaw = np.atan2(-mag_[1], mag_[0])

    return np.array([roll, pitch, yaw])