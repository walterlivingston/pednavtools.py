import numpy as np

def eul2rot(angles:np.array) -> np.array:
    C3 = np.array([[                 1,                  0,                  0],
                   [                 0,  np.cos(angles[0]),  np.sin(angles[0])],
                   [                 0, -np.sin(angles[0]),  np.cos(angles[0])]])
    C2 = np.array([[ np.cos(angles[1]),                  0, -np.sin(angles[1])],
                   [                 0,                  1,                  0],
                   [ np.sin(angles[1]),                  0,  np.cos(angles[1])]])
    C1 = np.array([[ np.cos(angles[2]),  np.sin(angles[2]),                  0],
                   [-np.sin(angles[2]),  np.cos(angles[2]),                  0],
                   [                 0,                  0,                  1]])
    return C1@C2@C3