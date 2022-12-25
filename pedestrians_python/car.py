from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

import intersection

class car:
    def __init__(self, mass = 1000, steer_limit = [-pi/4, pi/4], initial_state = [-25,-8.5,0], length = 4, width = 1.5):
        self.mass = mass
        self.steer_limit = steer_limit
        self.state = initial_state
        self.length = length
        self.width = width

    def initialize_plot(self):
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = plt.axes(xlim=(-25,0), ylim=(-25,0))
        self.line, = self.ax.plot([], [], marker = 'o')

    def compute_state(self):
        q0 = self.state
        q1 = self.state

        v = 2
        omega = 0
        dt = 0.1

        q1[0] = q0[0] + v*cos(q0[2])*dt
        q1[1] = q0[1] + v*sin(q0[2])*dt
        q1[2] = q0[2] + omega*dt

        self.state = q1

    def animate(self, i):

        self.compute_state()

        self.line.set_data(self.state[0], self.state[1])

        return self.line,

    def run_anim(self):
        self.initialize_plot()
        intersection.plot_env()
        plt.xlim(-25,0)
        plt.ylim(-25,0)

        anim = animation.FuncAnimation(self.fig, self.animate,
                                   frames=200, interval=20, blit=True)
        plt.show()

if __name__ == '__main__':
    car = car()
    car.run_anim()
