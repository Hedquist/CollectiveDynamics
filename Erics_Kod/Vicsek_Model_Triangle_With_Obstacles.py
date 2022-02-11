import numpy as np
from numpy import arctan2 as atan2, sin, cos
import matplotlib.pyplot as plt
from IPython import display
from tkinter import *
from tkinter import ttk
import time


res = 800  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3))) # Set height x width window
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res) # Place canvas with origin in x och y
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Parameters of the simulation
R_f = 25; # From 0 to 100
eta = 0.01 # From 0 to 1
N = 5  # Number of particles

T = 100000  # Simulation time
dt = 0.03  # Time step

# Physical parameters of the system
l = 100 # Size of box
V = 6 # Particle velocity
R = 4 # Radius of circle

# Initialization
x = np.random.rand(N) * 2 * l - l  # x coordinates # Generate numbers in interval [0,1)
y = np.random.rand(N) * 2 * l - l  # y coordinates
phi = np.random.rand(N) * 2 * np.pi  # orientations

# Parameters of obstacles
x_obstacles = [0.5*2*l-l, 0.5*2*l-l, 0.25*2*l-l, 0.75*2*l-l]
y_obstacles = [0.75*2*l-l, 0.25*2*l-l, 0.5*2*l-l, 0.5*2*l-l]
N_obstacles = len(x_obstacles)
R_f_obstacles = R_f-10
gamma0 = 10# Strength of change in orientation

particles = []
listR_f = []
obstacles = []

# Parameters of triangle
c = 2*R # Length of triangle
epsilon = np.pi/6 # Width of triangle angle

# Generate animated particles in Canvas
for j in range(N):
    theta = phi[j] # Store orientation
    # Convert to canvas coordinates and plot polygons
    particles.append(canvas.create_polygon((x[j]+c*np.cos(theta)+l) * res / l / 2,
                                           (y[j]+c*np.sin(theta)+l) * res / l / 2,
                                           (x[j]-c*np.cos(theta-epsilon)+l) * res / l / 2,
                                           (y[j]-c*np.sin(theta-epsilon)+l) * res / l / 2,
                                           (x[j]-c*np.cos(theta+epsilon)+l) * res / l / 2,
                                           (y[j]-c*np.sin(theta+epsilon)+l) * res / l / 2,
                                           outline=ccolor[0], fill=ccolor[0])) # x0,y0 - x1,y1
    # Plot influence radius
    listR_f.append(canvas.create_oval((x[j] - R_f + l) * res / l / 2,
                                      (y[j] - R_f + l) * res / l / 2,
                                      (x[j] + R_f + l) * res / l / 2,
                                      (y[j] + R_f + l) * res / l / 2,
                                      outline=ccolor[5])) # x0,y0 - x1,y1

# Generate animated obstacles in Canvas
for j in range(N_obstacles):
    # Create a circular obstacle
    obstacles.append(canvas.create_oval((x_obstacles[j] - R + l) * res / l / 2,
                                          (y_obstacles[j] - R + l) * res / l / 2,
                                          (x_obstacles[j] + R + l) * res / l / 2,
                                          (y_obstacles[j] + R + l) * res / l / 2,
                                          outline=ccolor[5],fill=ccolor[3])) # x0,y0 - x1,y1))
    # Create a obstacle interaction radius
    obstacles.append(canvas.create_oval((x_obstacles[j] + l) * res / l / 2,
                                          (y_obstacles[j]  + l) * res / l / 2,
                                          (x_obstacles[j]  + l) * res / l / 2,
                                          (y_obstacles[j]  + l) * res / l / 2,
                                          outline=ccolor[5])) # x0,y0 - x1,y1))

for i in range(T):

    x = (x + V * cos(phi) * dt + l) % (2 * l) - l  # Update x coordinates
    y = (y + V * sin(phi) * dt + l) % (2 * l) - l  # Update y coordinates
    x_y = np.array(list(zip(x, y))) # Append x and y to an array


    for j in range(N):
        n_0 = 0 # Number of obstacles in the radius
        h = 0
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)  # Calculate distances array to the particle
        interact = distances < R_f  # Create interaction indices, List of booleans
        for k in range(N_obstacles):
            distance_to_obstacle = np.sqrt((x_obstacles[k] - x[j]) ** 2 + (y_obstacles[k] - y[j]) ** 2)  # Calculate distance from particle to obstacle
            if(distance_to_obstacle < R_f) :
                n_0 += 1
                alpha_k = (np.arctan2(y[j], x[j]) - np.arctan2(y_obstacles[k] ,x_obstacles[k] ))
                h += gamma0/n_0*((np.sin(alpha_k - phi[j])))
        if(n_0 == 0):
            h = 0

        phi[j] = np.angle(np.sum(np.exp(phi[interact] * 1j))) + h + eta * np.random.randn()  # Update orientations
        # np.angle returns the angle of complex argument


    for j in range(N):
        # canvas.coords return the coordinate
        theta = phi[j]
        canvas.coords(particles[j],
                      (x[j] + c * np.cos(theta) + l) * res / l / 2,
                      (y[j] + c * np.sin(theta) + l) * res / l / 2,
                      (x[j] - c * np.cos(theta - epsilon) + l) * res / l / 2,
                      (y[j] - c * np.sin(theta - epsilon) + l) * res / l / 2,
                      (x[j] - c * np.cos(theta + epsilon) + l) * res / l / 2,
                      (y[j] - c * np.sin(theta + epsilon) + l) * res / l / 2)  # Updating animation coordinates
        canvas.coords(listR_f[j],
                      (x[j] - R_f + l) * res / l / 2,
                      (y[j] - R_f + l) * res / l / 2,
                      (x[j] + R_f + l) * res / l / 2,
                      (y[j] + R_f + l) * res / l / 2)  # Updating animation coordinates


    tk.title('t =' + str(round(i * dt * 100) / 100))  # Animation title
    tk.update()  # Update animation frame
    time.sleep(0.001)  # Wait between loops

Tk.mainloop(canvas)  # Release animation handle (close window to finish)