from bagpy import bagreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pednavtools as pnt
from pednavtools import ahrs

b = bagreader('data/11-28-23_nav_normal.bag')

vecnav_IMU = b.message_by_topic('/vectornav/IMU')
vecnav_Mag = b.message_by_topic('/vectornav/Mag')
ublox_GPS = b.message_by_topic('/ublox/enuVelocityTagged')
imu_data = pd.read_csv(vecnav_IMU)
mag_data = pd.read_csv(vecnav_Mag)
gps_data = pd.read_csv(ublox_GPS)

gps_time = gps_data['Time']
enu_vel = np.array([gps_data['enuVelocity.eastVelocity'], gps_data['enuVelocity.northVelocity'], gps_data['enuVelocity.upVelocity']])
course = np.arctan2(enu_vel[0,:], enu_vel[1,:])

imu_time = imu_data['Time']
acc = np.array([imu_data['linear_acceleration.x'], imu_data['linear_acceleration.y'], imu_data['linear_acceleration.z']])
gyr = np.array([imu_data['angular_velocity.x'], imu_data['angular_velocity.y'], imu_data['angular_velocity.z']])
mag = np.array([mag_data['magnetic_field.x'], mag_data['magnetic_field.y'], mag_data['magnetic_field.z']])
mag_time = mag_data['Time']

time, meas = pnt.time_sync(imu_time, mag_time)
course = np.interp(time, gps_time, course)
course[course < 0] = course[course < 0] + 2*np.pi
course = np.unwrap(course)
course = np.rad2deg(course)

raw_imu = pnt.IMU(acc,gyr,mag,imu_time,50,mag_time)
cal_imu = pnt.calibrate_mag(raw_imu)
cal_range = pnt.detect_still(cal_imu)
imu_n = pnt.body2nav(cal_imu, cal_range)
noise = pnt.find_characteristics(imu_n, cal_range)

noise.sigma_a = 0.7*noise.sigma_a
noise.sigma_g = 0.1*noise.sigma_g
noise.sigma_m = 0.5*noise.sigma_m

att = ahrs.smekf(imu_n, noise)
# att[...,2] = -att[...,2] + np.pi/2
att[..., att < 0] = att[..., att < 0] + 2*np.pi
att = -att + np.pi/2
att = np.unwrap(att)
att = np.rad2deg(att)

plt.plot(att[...,2], color='b', label='AHRS')
plt.plot(course, color='r', label='Course')
plt.show()

print('')