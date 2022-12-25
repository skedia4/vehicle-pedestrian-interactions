import matplotlib.pyplot as plt
import pypoman as ppm
import numpy as np

A = np.array([[-1,0],[1,0],[0,-1],[0,1]])
b_ll = np.array([[50],[-10],[50],[-10]])
b_lr = np.array([[4],[10],[50], [-10]])
b_ul = np.array([[50],[-10],[4],[10]])
b_ur = np.array([[4],[10],[4],[10]])

b_ll_wait1 = np.array([[50],[-10],[11],[-10]])
b_ll_wait2 = np.array([[11],[-10],[50],[-10]])
b_lr_wait1 = np.array([[4],[-3],[50], [-10]])
b_lr_wait2 = np.array([[4],[10],[11], [-10]])
b_ul_wait1 = np.array([[50],[-10],[4],[-3]])
b_ul_wait2 = np.array([[11],[-10],[4],[10]])
b_ur_wait1 = np.array([[4],[10],[4],[-3]])
b_ur_wait2 = np.array([[4],[-3],[4],[10]])

b_cross1 = np.array([[4],[-3],[7],[-4]])
b_cross2 = np.array([[11],[-10],[10],[-7]])
b_cross3 = np.array([[10],[-7],[4],[-3]])
b_cross4 = np.array([[7],[-4],[11],[-10]])



def plot_env():
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ll))
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_lr))
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ul))
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ur))

    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ll_wait1), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ll_wait2), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_lr_wait1), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_lr_wait2), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ul_wait1), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ul_wait2), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ur_wait1), color = 'yellow')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_ur_wait2), color = 'yellow')

    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_cross1), color = 'blue')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_cross2), color = 'blue')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_cross3), color = 'blue')
    ppm.polygon.plot_polygon(ppm.duality.compute_polytope_vertices(A, b_cross4), color = 'blue')


    plt.plot([-50,-10],[-7,-7],color = 'blue', linestyle = 'dashed')
    plt.plot([-7,-7],[-50,-10],color = 'blue', linestyle = 'dashed')
    plt.plot([-4,10],[-7,-7],color = 'blue', linestyle = 'dashed')
    plt.plot([-7,-7],[-4,10],color = 'blue', linestyle = 'dashed')



if __name__ == '__main__':
    plt.figure(figsize=(6, 6))
    plt.xlim([-25,0])
    plt.ylim([-25,0])
    plot_env()
    plt.show()
