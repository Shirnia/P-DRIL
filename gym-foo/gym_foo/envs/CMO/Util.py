from typing import Final
import numpy as np

Mach = 343


def getAngle(a, b, c):
    # returns the angle (in rads) of BA,BC

    # https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle


def absMin(x, y):
    arr = [x, y]
    idx = np.argmin(np.absolute(arr))
    return arr[idx]


def rss(a):
    s = np.square(a)
    ss = np.sum(s)
    rss = np.sqrt(ss)
    return rss


def cart2sph(cartesian):
    x = cartesian[0]
    y = cartesian[1]
    z = cartesian[2]

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return np.float64([az, el, r])


def sph2cart(spherical):
    az = spherical[0]
    el = spherical[1]
    r  = spherical[2]

    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return np.float64([x, y, z])