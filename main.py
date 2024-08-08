import numpy as np
from numpy import linalg as la
import quaternion as q
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
b = bagreader('data/11-28-23_nav_normal.bag')
vecnav_IMU = b.message_by_topic('/vectornav/IMU')
vecnav_Mag = b.message_by_topic('/vectornav/Mag')
print(b.topic_table)
ublox_GPS = b.message_by_topic('/ublox/llhPositionTagged')
ublox_GPS_vel = b.message_by_topic('/ublox/enuVelocityTagged')
imu_data = pd.read_csv(vecnav_IMU)
mag_data = pd.read_csv(vecnav_Mag)
gps_data = pd.read_csv(ublox_GPS_vel)

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

# n = max(imu.acc_n.shape) -2
# quat = np.empty([1, n-1], dtype=object)
# idx = 0
# for i in range(1, n):
#     quat[:,i-1] = ahrs.madgwick.filter(imu.acc_n[:,i-1], imu.gyr_n[:,i-1], mag=imu.mag_n[:,i-1], dt=dt, beta=beta, zeta=zeta)
#     # testq = np.quaternion(0, imu.mag_n[0,idx], imu.mag_n[1,idx], imu.mag_n[2,idx])
#     # if time_mag[idx] - time_acc[i] < 0 and testq.norm() < mag_th:
#     #     quat[:,i] = ahrs.madgwick.filter(imu.acc_n[:,i], imu.gyr_n[:,i], mag=imu.mag_n[:,idx], dt=dt, beta=beta, zeta=zeta)
#     # else:
#     #     quat[:,i] = ahrs.madgwick.filter(imu.acc_n[:,i], imu.gyr_n[:,i], dt=dt, beta=beta)
#     # if time_mag[idx] - time_acc[i] < 0:
#     #     idx = idx + 1

# att = np.zeros([3,n])
# for i in range(0,n):
#     att[:,i] = q.as_euler_angles(quat[:,i].item())

# plt.figure()
# plt.plot(np.transpose(att))
# plt.show()