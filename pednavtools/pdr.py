from dataclasses import dataclass
from .imu import IMU
from .utils import variance_detector
import numpy as np

@dataclass
class PDR:
    ned: np.array
    lla: np.array
    att: np.array
    time: np.array
    count_s: np.array
    freq_s: float

    def __init__(self, ned: np.array = [], lla: np.array = [], att: np.array = [], time: np.array = [], count_s: float = -1, freq_s: float = -1):
        self.ned = ned
        self.lla = lla
        self.att = att
        self.time = time
        self.count_s = count_s
        self.freq_s = freq_s

def detect_still(imu: IMU, M:int = 11, th:float = 0.01) -> range:
    acc = imu.acc
    buf = acc[...,0]
    N = acc.size
    start, stop = N, N
    for i in range(M+1, N-1):
        win = acc[:, (i-M):(i-1)]
        standing, buf = variance_detector(win, M, th, buf)
        if(standing and start == N):
            start = i
        elif(not standing and start != N):
            stop = i
            break

    return range(start, stop)