import numpy
from numpy.random import random_sample
import random
from math import pi, cos, sin
import pypoman as ppm

def random_start_and_goal(areas, v_range = [0, 2]):

    idxs = range(len(areas))
    idx = random.choice(idxs)

    x0_range, y0_range = areas.pop(idx)

    idxs = range(len(areas))
    idx = random.choice(idxs)
    xg_range, yg_range = areas.pop(idx)

    rand_x0_norm = random_sample()
    rand_y0_norm = random_sample()
    rand_xg_norm = random_sample()
    rand_yg_norm = random_sample()
    rand_v_norm = random_sample()
    rand_theta_norm = random_sample()

    rand_x0 = rand_x0_norm*(x0_range[1] - x0_range[0]) + x0_range[0]
    rand_y0 = rand_y0_norm*(y0_range[1] - y0_range[0]) + y0_range[0]
    rand_xg = rand_xg_norm*(xg_range[1] - xg_range[0]) + xg_range[0]
    rand_yg = rand_yg_norm*(yg_range[1] - yg_range[0]) + yg_range[0]
    rand_v = rand_v_norm*(v_range[1] - v_range[0]) + v_range[0]
    rand_theta = rand_theta_norm*(2*pi)

    rand_v_x = rand_v*cos(rand_theta)
    rand_v_y = rand_v*sin(rand_theta)

    start = [rand_x0, rand_y0, rand_v_x, rand_v_y]
    goal = [rand_xg, rand_yg, 0, 0]

    return start, goal

def random_ped_params():
    mass = random_sample()*50 + 50
    A = random_sample()*100
    b = random_sample()
    k = random_sample()*10 + 5
    sigma = random_sample()*10 + 1
    return mass, A, b, k, sigma

if __name__ == '__main__':
    inter_ll = ([-25,-10],[-25,-10])
    inter_ul = ([-25,-10],[-4,0])
    inter_lr = ([-4,0],[-25,-10])
    inter_ur = ([-4,0],[-4,0])

    areas = [inter_ll, inter_lr, inter_ul, inter_ur]

    print(random_start_and_goal(areas))
