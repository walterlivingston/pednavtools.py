import numpy as np

class IMU:
    def __init__(self, acc, gyr, mag, fs):
        # Raw Measurements
        self.raw_acc = acc
        self.raw_gyr = gyr
        self.raw_mag = mag
        self.fs = fs
        # Calibrated & Filtered Data
        self.acc_cf = self.raw_acc
        self.gyr_cf = self.raw_gyr
        self.mag_cf = self.raw_mag
        # Rotated, Calibrated, & Filtered Data
        self.acc_n = np.zeros([3,1])
        self.gyr_n = np.zeros([3,1])
        self.mag_n = np.zeros([3,1])
        print("IMU")

    def mag_cal(self):
        # Offset Correction
        off_x = (max(self.raw_mag[0,:]) + min(self.raw_mag[0,:])) / 2
        off_y = (max(self.raw_mag[1,:]) + min(self.raw_mag[1,:])) / 2
        off_z = (max(self.raw_mag[2,:]) + min(self.raw_mag[2,:])) / 2
        # Scale Correction
        dx_ = (max(self.raw_mag[0,:]) - min(self.raw_mag[0,:])) / 2
        dy_ = (max(self.raw_mag[1,:]) - min(self.raw_mag[1,:])) / 2
        dz_ = (max(self.raw_mag[2,:]) - min(self.raw_mag[2,:])) / 2
        del_ = (dx_ + dy_ + dz_) / 3
        if dx_ != 0 or dy_ != 0 or dz_ != 0:
            scale_x = del_ / dx_
            scale_y = del_ / dy_
            scale_z = del_ / dz_
        else:
            scale_x = 1
            scale_y = 1
            scale_z = 1

        self.mag_cf[0,:] = (self.raw_mag[0,:] - off_x) * scale_x
        self.mag_cf[1,:] = (self.raw_mag[1,:] - off_y) * scale_y
        self.mag_cf[2,:] = (self.raw_mag[2,:] - off_z) * scale_z
        print("Mag Cal")

    def body_to_nav(self, range = 0, th=0.01):
        # Initialize the Navigation Frame Measurements
        self.acc_n = self.acc_cf
        self.gyr_n = self.gyr_cf
        self.mag_n = self.mag_cf
        # Initial Euler Angle Estimates
        pitch = np.pi / 2
        roll = np.pi / 2
        yaw = 0
        # Iterative Calculation of Euler Angles to Drive Roll
        # and Pitch to near 0
        while pitch > th and roll > th:
            if self.acc_n.size > 3:
                acc_ = np.mean(self.acc_n[:,range],axis=1)
            else:
                acc_ = self.acc_n
            roll = np.arctan2(acc_[1], acc_[2]).item()
            pitch = np.arctan(acc_[0]/np.sqrt(acc_[1]**2 + acc_[2]**2)).item()

            C = self.DCM([roll, pitch, yaw])
            mag = C@self.mag_cf
            if mag.size > 3:
                mag_ = np.mean(mag[:, range],axis=1)
            else:
                mag_ = mag
            yaw = np.arctan2(mag_[1], mag_[0]).item()

            C = self.DCM([roll, pitch, yaw])

            self.acc_n = C@self.acc_n
            self.gyr_n = C@self.gyr_n
            self.mag_n = C@self.mag_n

        print("Body 2 Nav")

    def filter(self, freq):
        print("Filter")

    def DCM(self, angles):
        ct = np.cos(angles)
        st = np.sin(angles)

        cz = ct[0]   
        cx = ct[2]
        cy = ct[1]
        sx = st[2]
        sy = st[1]
        sz = st[0]

        return np.array([[cy*cz, sy*sx*cz - sz*cx, sy*cx*cz + sz*sx],
                         [cy*sz, sy*sx*sz + cz*cx, sy*cx*sz - cz*sx],
                         [  -sy,            cy*sx,            cy*cx]])