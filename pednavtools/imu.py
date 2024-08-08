import numpy as np
from conversions import *

def magCal(mag:np.array) -> np.array:
    # Offset Correction
    off_x = (max(mag[0,:]) + min(mag[0,:])) / 2
    off_y = (max(mag[1,:]) + min(mag[1,:])) / 2
    off_z = (max(mag[2,:]) + min(mag[2,:])) / 2
    # Scale Correction
    dx_ = (max(mag[0,:]) - min(mag[0,:])) / 2
    dy_ = (max(mag[1,:]) - min(mag[1,:])) / 2
    dz_ = (max(mag[2,:]) - min(mag[2,:])) / 2
    del_ = (dx_ + dy_ + dz_) / 3
    if dx_ != 0 or dy_ != 0 or dz_ != 0:
        scale_x = del_ / dx_
        scale_y = del_ / dy_
        scale_z = del_ / dz_
    else:
        scale_x = scale_y = scale_z = 1

    return np.vstack([(mag[0,:] - off_x) * scale_x, (mag[1,:] - off_y) * scale_y, (mag[2,:] - off_z) * scale_z])

def body2nav(acc:np.array, mag:np.array, range = 0, th=0.01) -> np.array:
    # Initialize the Navigation Frame Measurements
    acc_n = acc
    mag_n = mag
    # Initial Euler Angle Estimates
    rpy = np.array([np.pi/2, np.py/2, 0])
    # Iterative Calculation of Euler Angles to Drive Roll
    # and Pitch to near 0
    ret = np.eye(3)
    while rpy[0] > th and rpy[1] > th:
        acc_ = np.mean(acc_n[...,range],axis=1)
        rpy[0] = np.arctan2(acc_[1], acc_[2]).item()
        rpy[1] = np.arctan(acc_[0]/np.sqrt(acc_[1]**2 + acc_[2]**2)).item()

        C = eul2rot(rpy)
        mag_t = C@mag_n
        mag_ = np.mean(mag_t[..., range],axis=1)
        rpy[2] = np.arctan2(mag_[1], mag_[0]).item()

        C = eul2rot(rpy)
        acc_n = C@acc_n
        mag_n = C@acc_n
        ret = C@ret
    
    return ret