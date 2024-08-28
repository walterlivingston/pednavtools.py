import numpy as np

def skew(vec:np.array) -> np.array:
    return np.array([[      0, -vec[2],  vec[1]],
                     [ vec[2],       0, -vec[0]],
                     [-vec[1],  vec[0],       0]])

def time_sync(time1, time2):
    time = np.append(time1, time2)
    L = time1.size
    I = np.argsort(time)
    time = np.sort(time)
    meas = I > L

    return time - time[0], meas

def variance_detector(arr:np.array, win:int, th:float, buf:np.array):
    """Detect when buffer variance is below threshold
    
    Author: Howard Chen"""
    if win == 0:
        ln = 1
    else:
        sz = arr.shape
        ln = sz[1]

    if ln < win:
        buf = np.column_stack((buf, arr))
    else:
        buf = np.column_stack((buf[...,1:], arr))

    Sn = np.sqrt(buf[0,...]**2 + buf[1,...]**2 + buf[2,...]**2)

    if np.var(Sn) < th:
        return True, buf
    else:
        return False, buf