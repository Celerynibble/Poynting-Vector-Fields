# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 11:54:44 2025

@author: valde
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

epsilon_0 = 8.854 * 10**-12  #permittivity of free space
mu_0 = 4 * np.pi * 10**-7  #permeability of free space
q = 1  #charge
m = 1  #magnetic moment
x0, y0 = 0, 0  #dipole

x, y, z = sp.symbols('x y z')    

#position vector
r_vector = sp.Matrix([x - x0, y - y0, 0])  
r_magnitude = r_vector.norm()

#E-field from point charge
E_vector = (1 / (4 * sp.pi * epsilon_0)) * (q / r_magnitude**3) * r_vector

#Mag-field from dipole (XY)
m_vector = sp.Matrix([0, 0, m])  
r_hat = r_vector / r_magnitude  
B_vector_xy = (mu_0 / (4 * sp.pi)) * (1 / r_magnitude**3) * (3 * (m_vector.dot(r_hat)) * r_hat - m_vector)

#XZ
r_vector_xz = sp.Matrix([x - x0, 0, z - y0])  
r_magnitude_xz = r_vector_xz.norm()
r_hat_xz = r_vector_xz / r_magnitude_xz  
B_vector_xz = (mu_0 / (4 * sp.pi)) * (1 / r_magnitude_xz**3) * (3 * (m_vector.dot(r_hat_xz)) * r_hat_xz - m_vector)

#poynting vector
S_vector = E_vector.cross(B_vector_xy)

#here we define a function to normalize so the plot looks neat. 
#we only care about direction
def normalize(vector):
    return vector / (vector.norm() + 1e-12)  # Prevent division by zero

E_unit = normalize(E_vector)
B_unit_xy = normalize(B_vector_xy)
B_unit_xz = normalize(B_vector_xz)
S_unit = normalize(S_vector)

E_x = sp.lambdify((x, y), E_unit[0], 'numpy')
E_y = sp.lambdify((x, y), E_unit[1], 'numpy')
B_x = sp.lambdify((x, y), B_unit_xy[0], 'numpy')
B_y = sp.lambdify((x, y), B_unit_xy[1], 'numpy')
B_x_xz = sp.lambdify((x, z), B_unit_xz[0], 'numpy')
B_z_xz = sp.lambdify((x, z), B_unit_xz[2], 'numpy')
Sx = sp.lambdify((x, y), S_unit[0], 'numpy')
Sy = sp.lambdify((x, y), S_unit[1], 'numpy')

x_vals = np.linspace(-2, 2, 30)  # More points for denser field
y_vals = np.linspace(-2, 2, 30)
z_vals = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x_vals, y_vals)
X_xz, Z_xz = np.meshgrid(x_vals, z_vals)

Ex = E_x(X, Y)
Ey = E_y(X, Y)
Bx = B_x(X, Y)
By = B_y(X, Y)
Bx_xz_vals = B_x_xz(X_xz, Z_xz)
Bz_xz_vals = B_z_xz(X_xz, Z_xz)
Sx_vals = Sx(X, Y)
Sy_vals = Sy(X, Y)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.quiver(X, Y, Ex, Ey, scale=40, color='b', width= 0.002)
plt.scatter(x0, y0, color='r', s=100, zorder=5)
plt.title('Electric Field in XY Plane')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.quiver(X, Y, Bx, By, scale=40, color='r', width=0.002)
plt.scatter(x0, y0, color='b', s=100, zorder=5)
plt.title('Magnetic Field in XY Plane')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.quiver(X_xz, Z_xz, Bx_xz_vals, Bz_xz_vals, scale=40, color='r', width=0.002)
plt.scatter(x0, y0, color='b', s=100, zorder=5)
plt.title('Magnetic Field in XZ Plane')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.quiver(X, Y, Sx_vals, Sy_vals, scale=40, color='navy', width=0.002)  # Smaller arrows
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axis('equal')
plt.grid(False)
plt.title("Poynting Vector Field in XY Plane")
plt.show()

print()
