"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufactured soution using FEM
"""
import numpy as np
# import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from time import time

class Poisson_2d:

    def __init__(self, domain, u_bound_func, f_func):
        """
        Initialises methods to be used on the problem defined by the inputs.
        
        domain: dict with the following keys
            vertices: np.ndarray of shape (num of nodes, 2)
                each row with x,y coords of node_id(=row_id)
            vertex_markers: np.ndarray of shape (num of nodes, 1)
                entry 1 for if node_id(=row_id) is on the boundary; 0 otherwise
            triangles: np.ndarray of shape (num of elems, 3)
                each row with node ids of triangle
        u_bound_func: gives boundary value as output if input is a boundary point (x,y)
        f_func: forcing function gives output f(x,y)
        """
        self.vertex_markers = domain['vertex_markers'].reshape(-1)
        # self.vertex_known_id = self.vertex_markers.cumsum() - 1
        # self.vertex_unknown_id = (1 - self.vertex_markers).cumsum() - 1

        self.vert_known_list = np.where(self.vertex_markers == 1)[0].tolist()
        self.vert_unknown_list = np.where(self.vertex_markers == 0)[0].tolist()

        self.x = domain['vertices'][:,0]
        self.y = domain['vertices'][:,1]
        self.tri = domain['triangles']
        self.tri_flat = self.tri.flatten()

        self.triang = mtri.Triangulation(self.x, self.y, self.tri)  # to be used for plotting

        self.u_known = u_bound_func(self.x[self.vert_known_list], self.y[self.vert_known_list])
        self.n_known = len(self.vert_known_list)
        self.n_unknown = len(self.vert_unknown_list)
        self.n_elems = self.tri.shape[0]

        self.f = f_func

        self.time_logs = []


    def clock_on(self, message):
        self.time_logs.append([time(), message])

    def clock_off(self):
        t_beg = self.time_logs[-1][0]
        self.time_logs[-1][0] = time() - t_beg
        
    def get_K_f(self):
        """
        Calculating global stiffness matrix and consistent load vector
        considering 3 node triangles, using tested lines of code (BasicFEM).
        """
        self.clock_on("Forming K and f as dict")
        K_glob_dict = {}
        f_glob_dict = {}
        for e_id,vert_ids in enumerate(self.tri):
            x_vert = self.x[vert_ids]
            x02 = x_vert[0] - x_vert[2]
            x12 = x_vert[1] - x_vert[2]

            y_vert = self.y[vert_ids]
            y02 = y_vert[0] - y_vert[2]
            y12 = y_vert[1] - y_vert[2]

            detJ = x02*y12 - x12*y02
            B = np.array([[y12, -y02, -y12+y02],
                        [-x12, x02, x12-x02]]) / detJ
            K_elem = np.matmul(B.T,B) * detJ / 2

            x_mid = np.sum(x_vert) / 3
            y_mid = np.sum(y_vert) / 3
            f_mid = self.f(x_mid,y_mid)
            f_int = f_mid * detJ / 6

            for v_l_id,v_g_id in enumerate(vert_ids):
                if v_g_id in f_glob_dict:
                    f_glob_dict[v_g_id].append(f_int)
                else:
                    f_glob_dict[v_g_id] = [f_int]
                for u_l_id,u_g_id in enumerate(vert_ids):
                    if (v_g_id,u_g_id) in K_glob_dict:
                        K_glob_dict[(v_g_id,u_g_id)].append(K_elem[v_l_id,u_l_id])
                    else:
                        K_glob_dict[(v_g_id,u_g_id)] = [K_elem[v_l_id,u_l_id]]
        self.clock_off()

        self.K_check = K_glob_dict
        
        self.clock_on("Forming K and f the global quantities")
        K_glob = np.zeros((self.n_unknown,self.n_unknown))
        K_known = np.zeros((self.n_unknown,self.n_known))
        f_glob = np.zeros(self.n_unknown)
        for v_ind, v_id in enumerate(self.vert_unknown_list):
            for u_ind, u_id in enumerate(self.vert_unknown_list):
                if (v_id,u_id) in K_glob_dict:
                    # print(f"uu glod id ({v_id},{u_id}) ; loc ind ({v_ind},{u_ind})")
                    K_glob[v_ind,u_ind] = np.sum(K_glob_dict[(v_id,u_id)])
            for u_ind, u_id in enumerate(self.vert_known_list):
                    if (v_id,u_id) in K_glob_dict:
                        # print(f"uk glod id ({v_id},{u_id}) ; loc ind ({v_ind},{u_ind})")
                        K_known[v_ind,u_ind] = np.sum(K_glob_dict[(v_id,u_id)])
            # No need to add an if condition for v_ind in f_glob_dict
            f_glob[v_ind] = np.sum(f_glob_dict[v_id])
        f_glob -= np.dot(K_known,self.u_known)
        self.clock_off()

        return K_glob, f_glob

    def get_u_fem(self, K_glob, f_glob):
        """
        Calls linear system solver and assembles u_sol
        """
        self.clock_on("Calling linear solver")
        u_unknown = np.linalg.solve(K_glob,f_glob)        
        self.clock_off()

        self.clock_on("Assembling u_sol")
        u_sol = np.zeros(self.n_known + self.n_unknown)
        u_sol[self.vert_known_list] = self.u_known
        u_sol[self.vert_unknown_list] = u_unknown
        self.clock_off()

        return u_sol

    def sol_FEM(self):
        K_glob, f_glob = self.get_K_f()
        u_sol = self.get_u_fem(K_glob, f_glob)
        return u_sol

    def plot_on_grid(self, u_inp, title, fname, plot_with_lines = True):
        cplot = plt.tricontourf(self.triang, u_inp, levels = 100)
        plt.colorbar(cplot)
        if plot_with_lines:
            plt.triplot(self.triang, 'ko-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(fname)
        plt.close()
