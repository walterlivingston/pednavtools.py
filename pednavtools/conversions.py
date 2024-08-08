import numpy as np

def eul2rot(angles:np.array) -> np.array:
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