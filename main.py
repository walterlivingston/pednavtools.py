import numpy as np
from numpy import linalg as la
import quaternion
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt

import pednavtools as pnt
import pednavtools.ahrs as ahrs

# Gains & Thresholds
dt = (1/50)
beta = 0.1
zeta = 0.04
mag_th = 0.15

# Load Data
b = bagreader('data/11-28-23_nav_side.bag')
vecnav_IMU = b.message_by_topic('/vectornav/IMU')
vecnav_Mag = b.message_by_topic('/vectornav/Mag')
ublox_GPS = b.message_by_topic('/ublox/enuVelocityTagged')
imu_data = pd.read_csv(vecnav_IMU)
mag_data = pd.read_csv(vecnav_Mag)
gps_data = pd.read_csv(ublox_GPS)

enu_vel = np.array([gps_data['enuVelocity.eastVelocity'], gps_data['enuVelocity.northVelocity'], gps_data['enuVelocity.upVelocity']])
course = np.arctan2(enu_vel[0,:], enu_vel[1,:])

acc = np.array([imu_data['linear_acceleration.x'], imu_data['linear_acceleration.y'], imu_data['linear_acceleration.z']])
gyr = np.array([imu_data['angular_velocity.x'], imu_data['angular_velocity.y'], imu_data['angular_velocity.z']])
mag = np.array([mag_data['magnetic_field.x'], mag_data['magnetic_field.y'], mag_data['magnetic_field.z']])

time_acc = imu_data['Time']
time_mag = mag_data['Time']
# Setup Sensor
imu = pnt.IMU(acc, gyr, mag, 0)
imu.mag_cal()
imu.body_to_nav(range(331))

n = max(imu.acc_n.shape) -2
quat = np.empty([1, n], dtype=object)
idx = 0
for i in range(0, n):
    if time_mag[idx] - time_acc[i] < 0 and la.norm(imu.mag_n[:,idx]) < mag_th:
        quat[:,i] = ahrs.madgwick.filter(imu.acc_n[:,i], imu.gyr_n[:,i], mag=imu.mag_n[:,idx], dt=dt, beta=beta, zeta=zeta)
        idx = idx + 1
    else:
        quat[:,i] = ahrs.madgwick.filter(imu.acc_n[:,i], imu.gyr_n[:,i], dt=dt, beta=beta)

att = np.zeros([3,n])
for i in range(0,n):
    att[:,i] = quaternion.as_euler_angles(quat[:,i].item())

plt.figure()
plt.plot(np.transpose(att))
plt.show()