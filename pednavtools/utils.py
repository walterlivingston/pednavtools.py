import numpy as np

def skew(vec:np.array) -> np.array:
    return np.array([[      0, -vec[2],  vec[1]],
                     [ vec[2],       0, -vec[0]],
                     [-vec[1],  vec[0],       0]])

def time_sync(time1, time2):
    N = max(len(time1), len(time2))
    bigN = len(time1) + len(time2)

    idx1, idx2 = 1,1
    meas = np.full([bigN], False)
    time = np.empty([bigN])
    for i in range(1,2*N-1):
        lenFlag1 = idx1 > len(time1)-1
        lenFlag2 = idx2 > len(time2)-1

        if not lenFlag1:
            changeFlag1 = (time1[idx1] == time1[idx1-1])
        else:
            changeFlag1 = True
        
        if not lenFlag2:
            changeFlag2 = (time2[idx2] == time2[idx2-1])
        else:
            changeFlag2 = True

        if changeFlag1 and changeFlag2:
            break

        if (idx2-1) < len(time2) and (idx1-1) < len(time1):
            timeFlag = time2[idx2-1] < time1[idx1-1]
            if timeFlag:
                meas[i-1] = True
                time[i-1] = time2[idx2-1]
                idx2 += 1
            else:
                meas[i-1] = False
                time[i-1] = time1[idx1-1]
                idx1 += 1
        elif (idx2-1) < len(time2):
            meas[i-1] = True
            time[i-1] = time2[idx2-1]
            idx2 += 1
        else:
            meas[i-1] = False
            time[i-1] = time1[idx1-1]
            idx1 += 1
    
    time, meas = np.squeeze(time), np.squeeze(meas)
    return time, meas

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