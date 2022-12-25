from math import pi, sin, cos,sqrt,exp
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib import animation

#import intersection

class ped:
    def __init__(self, initial_state = [-20,-11,1,1], goal_state= [-20, -1, 0, 0], mass = 62, A = 100, b = 1, k = 10, sigma = 1):
        self.mass = mass

        # initialization
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.v0 = sqrt(initial_state[2]**2 + initial_state[3]**2)

        # ped reaction info
        self.A = A
        self.b = b
        self.k = k
        self.sigma = sigma

        self.state = self.initial_state

    def compute_force(self, car_loc = (-10, -7)):
        k_des = self.k
        sigma_des = self.sigma

        v_d = [self.v0*(self.goal_state[0] - self.state[0])/(sqrt((self.goal_state[0] - self.state[0])**2 + (self.goal_state[1] - self.state[1])**2 + sigma_des**2)),
               self.v0*(self.goal_state[1] - self.state[1])/(sqrt((self.goal_state[0] - self.state[0])**2 + (self.goal_state[1] - self.state[1])**2 + sigma_des**2))]

        f_des = [k_des*(v_d[0]- self.state[2]), k_des*(v_d[1]-self.state[3])]

        A = self.A
        b = self.b
        d_v2p = sqrt((car_loc[0] - self.state[0])**2 + (car_loc[1] - self.state[1])**2)

        n_v2p = [(-(car_loc[0] - self.state[0]))/d_v2p, (-(car_loc[1] - self.state[1])-2)/d_v2p]
        f_veh = [n_v2p[0]*A*exp(-b*d_v2p), n_v2p[1]*A*exp(-b*d_v2p)]

        f_total = [f_des[0] + f_veh[0], f_des[1] + f_veh[1]]

        return f_total

    # def initialize_plot(self):
    #     self.fig = plt.figure(figsize=(6, 6))
    #     self.ax = plt.axes(xlim=(-25,0), ylim=(-25,0))
    #     self.line, = self.ax.plot([], [], marker = 'o')

    def compute_state(self, car_loc = (-10,-7), dt=0.1):
        x0,y0,vx0,vy0 = self.state

        f = self.compute_force(car_loc)

        vx1 = vx0 + f[0]*dt
        vy1 = vy0 + f[1]*dt

        x1 = x0 + vx0*dt
        y1 = y0 + vy0*dt

        self.state = [x1,y1,vx1,vy1]

    # def animate(self, i):
    #     self.compute_state()

    #     self.line.set_data(self.state[0], self.state[1])

    #     return self.line,

    # def run_anim(self):

    #     self.initialize_plot()
    #     intersection.plot_env()
    #     plt.xlim(-25,0)
    #     plt.ylim(-25,0)

    #     anim = animation.FuncAnimation(self.fig, self.animate,
    #                                frames=200, interval=100, blit=True)
    #     plt.show()

if __name__ == '__main__':
    car = ped()
    car.run_anim()
