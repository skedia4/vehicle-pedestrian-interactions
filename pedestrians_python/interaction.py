from math import pi, sin, cos,sqrt,exp,floor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from numpy.random import random_sample

import intersection
import ped
import car
from sampling import random_start_and_goal, random_ped_params

class scenario:
    def __init__(self, initial_states = [[-20, -11, 0, 0]], goal_states = [[-20, -1, 0, 0]], masses = [62], As = [100], bs = [1], ks = [10], sigmas = [1]):
        self.ped_num = len(initial_states)
        self.peds = [ped.ped(initial_states[i], goal_states[i], masses[i], As[i], bs[i], ks[i], sigmas[i]) for i in range(self.ped_num)]
        self.car = car.car()

    def initialize_plot(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(xlim=(-25,0), ylim=(-25,0))
        self.ped_states, = self.ax.plot([], [], marker = 'o', linestyle = "None")
        self.car_states, = self.ax.plot([], [], marker = '*')

    def animate(self, i):
        self.car.compute_state()
        self.car_states.set_data(self.car.state[0], self.car.state[1])

        x_peds = []
        y_peds = []
        for ped in self.peds:
            # Update closest point calculation for car
            car_loc = (self.car.state[0] + 1, self.car.state[1]-1)
            ped.compute_state(car_loc)
            x_peds.append(ped.state[0])
            y_peds.append(ped.state[1])

        self.ped_states.set_data(x_peds, y_peds)


        return self.ped_states, self.car_states

    def run_anim(self):

        self.initialize_plot()
        intersection.plot_env()
        plt.xlim(-25,0)
        plt.ylim(-25,0)

        anim = animation.FuncAnimation(self.fig, self.animate,
                                   frames=100, interval=100, blit=True)
        # plt.show()
        f = r"animation.gif"
        writergif = animation.PillowWriter(fps=30)
        anim.save(f, writer=writergif)

if __name__ == '__main__':

    inter_ll = ([-25,-10],[-25,-10])
    inter_ul = ([-25,-10],[-4,0])
    inter_lr = ([-4,0],[-25,-10])
    inter_ur = ([-4,0],[-4,0])

    areas = [inter_ll, inter_lr, inter_ul, inter_ur]

    initial_states = []
    goal_states = []
    masses = []
    As = []
    bs = []
    ks = []
    sigmas = []

    ped_num = floor(random_sample()*30) + 20
    for i in range(ped_num):
        areas = [inter_ll, inter_lr, inter_ul, inter_ur]
        start, goal = random_start_and_goal(areas)
        mass, A, b, k, sigma = random_ped_params()
        initial_states.append(start)
        goal_states.append(goal)
        masses.append(mass)
        As.append(A)
        bs.append(b)
        ks.append(k)
        sigmas.append(sigma)

    # initial_states = [[-20,-11,1,1], [-19,-11,1,1], [-12,-11,0,0.75]]
    # goal_states = [[-20, -1, 0, 0], [-21, -1, 0, 0], [0,0,1,1]]
    # masses = [62,62,62]
    # As = [100,100,100]
    # bs = [1,1,1]
    # ks = [10,10,10]
    # sigmas = [1,1,1]


    car = scenario(initial_states, goal_states, masses, As, bs, ks, sigmas)
    car.run_anim()
