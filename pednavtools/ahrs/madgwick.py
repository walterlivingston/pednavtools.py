import numpy as np
import quaternion

q_est = np.quaternion(1,0,0,0)
def filter(acc, gyr, dt, mag=None, beta=0.1, zeta=0):
    global q_est
    q_est_dot = q_est
    q_w = np.quaternion(0,gyr[0],gyr[1],gyr[2])
    q_w = q_w.normalized()
    q_a = np.quaternion(0,acc[0],acc[1],acc[2])
    q_a = q_a.normalized()

    Fg = np.zeros([3,1])
    Jg = np.zeros([3,4])
    grad = np.quaternion(1,0,0,0)

    Fg[0] = 2*(q_est.x*q_est.z - q_est.w*q_est.y) - q_a.x
    Fg[1] = 2*(q_est.w*q_est.x + q_est.y*q_est.z) - q_a.y
    Fg[2] = 2*(0.5 - q_est.x**2 - q_est.y**2) - q_a.z

    Jg[0,0] = -2*q_est.y
    Jg[0,1] =  2*q_est.z
    Jg[0,2] = -2*q_est.w
    Jg[0,3] =  2*q_est.x

    Jg[1,0] =  2*q_est.x
    Jg[1,1] =  2*q_est.w
    Jg[1,2] =  2*q_est.z
    Jg[1,3] =  2*q_est.y

    Jg[2,0] =  0
    Jg[2,1] = -4*q_est.x
    Jg[2,2] = -4*q_est.y
    Jg[2,3] =  0

    if mag is not None:
        q_m = np.quaternion(0, mag[0], mag[1], mag[2])
        q_m = q_m.normalized()
        Fb = np.zeros([3,1])
        Jb = np.zeros([3,4])

        h = q_est*q_m*q_est.conjugate()
        b = np.quaternion(0, np.sqrt(h.x**2 + h.y**2), 0, h.z)

        Fb[0] = 2*b.x*(0.5 - q_est.y**2 - q_est.z**2) + 2*b.z*(q_est.x*q_est.z - q_est.w*q_est.y) - q_m.x
        Fb[1] = 2*b.x*(q_est.x*q_est.y - q_est.w*q_est.z) + 2*b.z*(q_est.w*q_est.x + q_est.y*q_est.z) - q_m.y
        Fb[2] = 2*b.x*(q_est.w*q_est.y + q_est.x*q_est.z) + 2*b.z*(0.5 - q_est.x**2 - q_est.y**2) - q_m.z

        Jb[0,0] = -2*b.z*q_est.y
        Jb[0,1] =  2*b.z*q_est.z
        Jb[0,2] = -4*b.x*q_est.y - 2*b.z*q_est.w
        Jb[0,3] = -4*b.x*q_est.x + 2*b.z*q_est.x

        Jb[1,0] = -2*b.x*q_est.z + 2*b.z*q_est.x
        Jb[1,1] =  2*b.x*q_est.y + 2*b.z*q_est.w
        Jb[1,2] =  2*b.x*q_est.x + 2*b.z*q_est.z
        Jb[1,3] = -2*b.x*q_est.w + 2*b.z*q_est.y

        Jb[2,0] =  2*b.x*q_est.y
        Jb[2,1] =  2*b.x*q_est.z - 4*b.z*q_est.x
        Jb[2,2] =  2*b.x*q_est.w - 4*b.z*q_est.y
        Jb[2,3] =  2*b.x*q_est.x

        F = np.concatenate((Fg, Fb))
        J = np.concatenate((Jg, Jb))
    else:
        F = Fg
        J = Jg

    grad_vec = np.transpose(J)@F
    grad = np.quaternion(grad_vec[0], grad_vec[1], grad_vec[2], grad_vec[3])
    grad = grad.normalized()

    q_wc = 2*q_est*grad
    q_wc = q_w - zeta*q_wc
    q_est_dot = 0.5*q_est*q_wc

    q_est_dot = q_est_dot - beta*grad
    q_est = q_est + q_est_dot*dt
    q_est = q_est.normalized()
    return q_est