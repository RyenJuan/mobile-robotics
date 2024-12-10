import math
from math import hypot
import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2):
    """Euclidean distance between two points."""
    x1, y1 = p1
    x2, y2 = p2
    return hypot(x2 - x1, y2 - y1)

def data_restructuring(spline_points, cs_x, cs_y, t_fine, acc_max=2000, v_max=500):
    x, y = list(spline_points[0]), list(spline_points[1])
    v, va, vb, vc, running_dist = [], [], [], [], []

    dx_dt = cs_x.derivative()(t_fine)
    dy_dt = cs_y.derivative()(t_fine)
    d2x_dt2 = cs_x.derivative(2)(t_fine)
    d2y_dt2 = cs_y.derivative(2)(t_fine)

    numerator = (dx_dt**2 + dy_dt**2)**(3/2)
    denominator = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)  # Curvature formula for radius
    curvature = denominator / numerator

    """Calculate distance between each spline point"""
    # Restructuring spline_points from np.arrays to lists
    new_coords = [(x, y) for x, y in spline_points]
    res = [((new_coords[i]), (new_coords[i + 1])) for i in range(len(new_coords) - 1)]
    dist = [distance(point[0], point[1]) for point in res]

    for i in range(len(dist)):
        temp = sum(dist[:i])
        running_dist.append(temp)

    for i in range(len(spline_points)-1):
        if acc_max <= (v_max**2) * curvature[i]:
            va.append(math.sqrt(acc_max/curvature[i]))
        else:
            va.append(v_max)

        if running_dist[i] < (0.5*acc_max*(v_max/acc_max)**2):
            vb.append(running_dist[i]/((0.5*acc_max*(v_max/acc_max)**2))*v_max)
        else:
            vb.append(v_max + 1)

        if running_dist[i] > (running_dist[-1] - (0.5*acc_max*(v_max/acc_max)**2)):
            vc.append(((running_dist[-1] - running_dist[i])/(0.5*acc_max*(v_max/acc_max)**2))*v_max)
        else:
            vc.append(v_max + 1)


        v.append(min(va[i], vb[i], vc[i]))

    return list(x), list(y), running_dist, curvature, v


def time_calculator(x, y, running_dist, curvature, v):
    t = [0]
    v = [i for i in v if i != 0]
    for i in range(1, len(v)-1):
        t.append(t[i-1] + (running_dist[i] - running_dist[i-1])/v[i-1])
    return t


if __name__ == "__main__":
    pass



