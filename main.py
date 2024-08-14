import bagpy
from bagpy import bagreader
import pandas as pd
import numpy as np
import pednavtools as pnt

b = bagreader('data/11-28-23_nav_normal.bag')

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

raw_imu = pnt.IMU(acc,gyr,mag)
cal_imu = pnt.calibrate_mag(raw_imu)
cal_range = pnt.detect_still(cal_imu)
imu_n = pnt.body2nav(cal_imu, cal_range)
noise = pnt.find_characteristics(imu_n, cal_range)

print(noise)