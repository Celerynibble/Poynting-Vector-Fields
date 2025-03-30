# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 12:10:54 2025

@author: valde
"""

import numpy as np
import matplotlib.pyplot as plt

#parameters of the circuit
L = 1.0  #side length of the square circuit
lambda_0 = 0.1  #considering a linear charge variation, value at start pos.
epsilon_0 = 8.85e-12  #permittivity of free space
dipole_p = np.array([1,0])  #dipole to pretend as battery
mu_0 = 4 * np.pi * 1e-7  #permeability of free space
I = 1.0  #current in the wire

def electric_field_line_segment(start, end, lambda_start, lambda_end, grid_x, grid_y):
    E_x, E_y = np.zeros_like(grid_x), np.zeros_like(grid_y)
    num_points = 200  #number of discrete chunks field is computed for
    
    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    lambda_vals = np.linspace(lambda_start, lambda_end, num_points)
    dl = np.sqrt((x_vals[1] - x_vals[0])**2 + (y_vals[1] - y_vals[0])**2)
    
    for i in range(num_points):
        x_q, y_q, lambda_q = x_vals[i], y_vals[i], lambda_vals[i]
        dq = lambda_q*dl
        
        rx = grid_x - x_q
        ry = grid_y - y_q
        r = np.sqrt(rx**2 + ry**2)
        r[r < 1e-6] = 1e-6  
        
        dE = (dq/(4*np.pi*epsilon_0*r**3))
        E_x = E_x + dE*rx
        E_y = E_y + dE*ry
    
    return E_x, E_y

def dipole_field(px, py, grid_x, grid_y):
    p = dipole_p
    rx = grid_x - px
    ry = grid_y - py
    r = np.sqrt(rx**2 + ry**2)
    r[r < 1e-6] = 1e-6  
    
    dot_pr = p[0]*rx + p[1]*ry
    k = (1/(4*np.pi*epsilon_0*r**5))
    
    E_x = k*(3*dot_pr*rx - p[0]*r**2)
    E_y = k*(3*dot_pr*ry - p[1]*r**2)
    
    return E_x, E_y

def magnetic_field_wire_segment(start, end, grid_x, grid_y):
    B_z = np.zeros_like(grid_x)
    num_points = 200
    
    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    dl_x = np.gradient(x_vals)
    dl_y = np.gradient(y_vals)
    
    for i in range(num_points):
        x_q, y_q = x_vals[i], y_vals[i]
        dl = np.array([dl_x[i], dl_y[i]])
        
        rx = grid_x - x_q
        ry = grid_y - y_q
        r = np.sqrt(rx**2 + ry**2)
        r[r < 1e-6] = 1e-6  
        
        dB_z = (mu_0*I/(4*np.pi*r**3))*(dl[0]*ry - dl[1]*rx)
        B_z = B_z + dB_z
    
    return B_z


x = np.linspace(-1.5, 1.5, 40)
y = np.linspace(-1.5, 1.5, 40)
X, Y = np.meshgrid(x, y)

E_total_x, E_total_y = np.zeros_like(X), np.zeros_like(Y)
B_total_z = np.zeros_like(X)

top_left, top_right = (-L/2, L/2), (L/2, L/2)
bottom_left, bottom_right = (-L/2, -L/2), (L/2, -L/2)

for (start, end) in [(top_left, top_right), (top_right, bottom_right),
                     (bottom_right, bottom_left), (bottom_left, top_left)]:
    E_x, E_y = electric_field_line_segment(start, end, lambda_0, -lambda_0, X, Y)
    B_z = magnetic_field_wire_segment(start, end, X, Y)
    E_total_x = E_total_x + E_x
    E_total_y = E_total_y + E_y
    B_total_z = B_total_z + B_z

E_dx, E_dy = dipole_field(0, L/2, X, Y)
E_total_x = E_total_x + E_dx
E_total_y = E_total_y + E_dy

def normalize_vectors(Vx, Vy):
    magnitude = np.sqrt(Vx**2 + Vy**2)
    magnitude[magnitude < 1e-9] = 1e-9
    return Vx/magnitude, Vy/magnitude


#poynting vector
S_x = E_total_y*B_total_z
S_y = -E_total_x*B_total_z

E_total_x, E_total_y = normalize_vectors(E_total_x, E_total_y)
S_x, S_y = normalize_vectors(S_x, S_y)

sample_step = 5
sample_X = X[::sample_step, ::sample_step]
sample_Y = Y[::sample_step, ::sample_step]
sample_B_z = B_total_z[::sample_step, ::sample_step]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].quiver(X, Y, E_total_x, E_total_y, color='blue', scale=40)
axes[0].set_title("Electric Field")

for i in range(sample_X.shape[0]):
    for j in range(sample_X.shape[1]):
        if sample_B_z[i, j] > 0:
            axes[1].scatter(sample_X[i, j], sample_Y[i, j], color='red', marker='o', label='B out of plane' if i == 0 and j == 0 else "")
        elif sample_B_z[i, j] < 0:
            axes[1].scatter(sample_X[i, j], sample_Y[i, j], color='red', marker='x', label='B into plane' if i == 0 and j == 0 else "")
axes[1].set_title("Magnetic Field")

axes[2].quiver(X, Y, S_x, S_y, color='green', scale=40)
axes[2].set_title("Poynting Vector Field")

for ax in axes:
    ax.plot([-L/2, L/2, L/2, -L/2, -L/2], [L/2, L/2, -L/2, -L/2, L/2], 'k-', linewidth=2)
    ax.scatter(0, L/2, color='black', marker='o', s=50, label='electric cell')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.grid()

plt.show()
