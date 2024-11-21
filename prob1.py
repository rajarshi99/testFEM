"""
Poisson problem: -laplacian(u)(x,y) = f(x,y)
Method of manufacture soution using FEM
Here we define
- u (unknown function)
- f (forcing function)
- x and y the coordinates of the points
"""
import numpy as np
import triangle as tr

from poisson_2d import Poisson_2d

import sys

def u(x,y):
    # return np.where(y == 1, 1, 0)
    # return np.sin(np.pi*x)*np.sin(np.pi*y)
    # return x**2*y**2
    return x*y

def f(x,y):
    # return 0
    # return 2*(np.pi)**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    # return -2*(x**2 + y**2)
    return 0

try:
    num_points = int(sys.argv[1])
except:
    num_points = 4

output_dir = "trial/"                           # folder to save the plots 

x_1d = np.linspace(-1,1,num_points, dtype=np.float32)
y_1d = np.linspace(-1,1,num_points, dtype=np.float32)
x,y = np.meshgrid(x_1d,y_1d)
x = x.flatten()
y = y.flatten()

## added stuff
domain = {'vertices' : np.column_stack((x,y))}
for key, val in tr.triangulate(domain).items():
    domain[key] = val
    print(key)

p_2d = Poisson_2d(domain, u, f)

u_exct = u(x,y)
p_2d.plot_on_mesh(u_exct, f"Exact solution {num_points}", f"{output_dir}u_exct_{num_points}.png")
u_sol = p_2d.sol_FEM()
p_2d.plot_on_mesh(u_sol, f"FEM solution {num_points}", f"{output_dir}u_fem_{num_points}.png")
p_2d.plot_on_mesh(u_exct - u_sol, f"Exact solution - FEM solution {num_points}", f"{output_dir}u_err_{num_points}.png")

print(p_2d.time_logs)

