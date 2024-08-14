import numpy as np
import quaternions as q

def mag_madgwick(q_est:np.array, mag:np.array):
    q_m = np.array([0, mag[0], mag[1], mag[2]])
    q_m = q.qNormalize(q_m)
    h = q.qMult(q_est, q.qMult(q_m, q.qConj(q_est)))
    b = np.array([0, np.sqrt(h[1]**2 + h[2]**2), 0, h[3]])
    return b