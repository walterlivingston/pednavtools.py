import numpy as np
from .conversions import eul2rot
from dataclasses import dataclass

@dataclass
class IMU:
    """Class for managing inertial measurements"""
    acc: np.array
    gyr: np.array
    mag: np.array
    time: np.array
    fs: float

    def __init__(self, acc: np.array = [], gyr: np.array = [], mag: np.array = [], time: np.array = [], fs: float = -1):
        self.acc = acc
        self.gyr = gyr
        self.mag = mag
        self.time = time
        self.fs = fs

@dataclass
class Noise:
    sigma_a: np.array
    sigma_g: np.array
    sigma_m: np.array
    bias_a: np.array
    bias_g: np.array
    bias_m: np.array

    def __init__(self, sigma_a:np.array = [], sigma_g:np.array = [], sigma_m:np.array = [], bias_a:np.array = [], bias_g:np.array = [], bias_m:np.array = []):
        self.sigma_a = sigma_a
        self.sigma_g = sigma_g
        self.sigma_m = sigma_m
        self.bias_a = bias_a
        self.bias_g = bias_g
        self.bias_m = bias_m

def calibrate_mag(imu:IMU) -> IMU:
    mag = imu.mag
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

    mag_c = np.vstack([(mag[0,:] - off_x) * scale_x, (mag[1,:] - off_y) * scale_y, (mag[2,:] - off_z) * scale_z])
    return IMU(imu.acc, imu.gyr, mag_c, imu.time, imu.fs)

def body2nav(imu:IMU, rng: range = range(1), th=0.01) -> IMU:
    # Initialize the Navigation Frame Measurements
    acc_n = imu.acc
    gyr_n = imu.gyr
    mag_n = imu.mag
    # Initial Euler Angle Estimates
    rpy = np.array([np.pi/2, np.pi/2, 0])
    # Iterative Calculation of Euler Angles to Drive Roll
    # and Pitch to near 0
    ret = np.eye(3)
    while rpy[0] > th and rpy[1] > th:
        acc_ = np.mean(acc_n[...,rng],axis=1)
        rpy[0] = np.arctan2(acc_[1], acc_[2]).item()
        rpy[1] = np.arctan(acc_[0]/np.sqrt(acc_[1]**2 + acc_[2]**2)).item()

        C = eul2rot(rpy)
        mag_t = C@mag_n
        mag_ = np.mean(mag_t[..., rng],axis=1)
        rpy[2] = np.arctan2(mag_[1], mag_[0]).item()

        C = eul2rot(rpy)
        acc_n = C@acc_n
        gyr_n = C@gyr_n
        mag_n = C@mag_n

    ret = IMU(acc_n, gyr_n, mag_n, imu.time, imu.fs)
   
    return ret

def find_characteristics(imu:IMU, rng:range = range(1)):
    sigma_a = np.std(imu.acc[...,rng],axis=1)
    sigma_g = np.std(imu.gyr[...,rng],axis=1)
    sigma_m = np.std(imu.mag[...,rng],axis=1)
    
    bias_a = np.mean(imu.acc[...,rng],axis=1)
    bias_g = np.mean(imu.gyr[...,rng],axis=1)
    bias_m = np.mean(imu.mag[...,rng],axis=1)

    return Noise(sigma_a, sigma_g, sigma_m, bias_a, bias_g, bias_m)