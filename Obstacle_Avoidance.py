import numpy as np
from numpy import arctan2 as atan2, sin, cos
import matplotlib.pyplot as plt
from IPython import display
from scipy.constants import Boltzmann as kB
from tkinter import *
from tkinter import ttk
from PIL import ImageGrab
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import time
from shapely.geometry import Polygon


res = 800  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3))) # Set height x width window
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res) # Place canvas with origin in x och y
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Parameters of the simulation
R_interaction = 25; # From 0 to 100
eta = 0.05# From 0 to 1
N = 5 # Number of particles

T = 100000  # Simulation time
dt = 0.03  # Time step

# Physical parameters of the system
l = 100 # Size of box
V = 10 # Particle velocity
R = 4 # Radius of circle

# Initialization
x = np.random.rand(N) * 2 * l - l  # x coordinates # Generate numbers in interval [0,1)
y = np.random.rand(N) * 2 * l - l  # y coordinates
phi = np.random.rand(N) * 2 * np.pi  # orientations


particles = []
listR_f = []
list_aheadarrows=[]
list_ciruclar_obstacles=[]
list_rectangular_obstacles=[]

# Parameters of triangle
c = 1.5*R # Length of triangle
epsilon = np.pi/12 # Width of triangle angle
I = 5 # Inertia
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
    listR_f.append(canvas.create_oval((x[j] - R_interaction + l) * res / l / 2,
                                      (y[j] - R_interaction + l) * res / l / 2,
                                      (x[j] + R_interaction + l) * res / l / 2,
                                      (y[j] + R_interaction + l) * res / l / 2,
                                      outline=ccolor[5])) # x0,y0 - x1,y1
    list_aheadarrows.append((x[j]+R_interaction*np.cos(phi[j]),y[j] + R_interaction*np.sin(phi[j])))


def wall_avoidance_angle(list_aheadarrows):
    if (list_aheadarrows[0] > l):
        distance = x[j] - l
        force = 1 / (distance ** 2 + 1)
        torque = force * R_interaction
        angular_velocity = 2 * torque / I
        cross_product = np.cross([np.cos(phi[j]), np.sin(phi[j])], [-1, 0])
        angle = angular_velocity * dt * np.sign(cross_product)
    elif (list_aheadarrows[0] < -l):
        distance = x[j] + l
        force = 1 / (distance ** 2 + 1)
        torque = force * R_interaction
        angular_velocity = 2 * torque / I
        cross_product = np.cross([cos(phi[j]), np.sin(phi[j])], [1, 0])
        angle = angular_velocity * dt * np.sign(cross_product)
    elif (list_aheadarrows[1] > l):
        distance = y[j] - l
        force = 1 / (distance ** 2 + 1)
        torque = force * R_interaction
        angular_velocity = 2 * torque / I
        cross_product = np.cross([cos(phi[j]), np.sin(phi[j])], [0, -1])
        angle = angular_velocity * dt * np.sign(cross_product)
    elif (list_aheadarrows[1] < -l):
        distance = y[j] + l
        force = 1 / (distance ** 2 + 1)
        torque = force * R_interaction
        angular_velocity = 2 * torque / I
        cross_product = np.cross([cos(phi[j]), np.sin(phi[j])], [0, 1])
        angle = angular_velocity * dt * np.sign(cross_product)
    else:
        angle = 0
    return angle


for i in range(T):

    x = (x + V * cos(phi) * dt )    # Update x coordinates
    y = (y + V * sin(phi) * dt )   # Update y coordinates


    for j in range(N):
        list_aheadarrows[j] = ((x[j] + R_interaction * np.cos(phi[j]), y[j] + R_interaction * np.sin(phi[j])))
        angle = wall_avoidance_angle(list_aheadarrows[j])
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)  # Calculate distances array to the particle
        interact = distances < R_interaction  # Create interaction indices, List of booleans
        phi[j] = np.angle(np.sum(np.exp(phi[interact] * 1j)))  + angle + eta * np.random.randn()  # Update orientations
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
                      (x[j] - R_interaction + l) * res / l / 2,
                      (y[j] - R_interaction + l) * res / l / 2,
                      (x[j] + R_interaction + l) * res / l / 2,
                      (y[j] + R_interaction + l) * res / l / 2)  # Updating animation coordinates


    tk.title('t =' + str(round(i * dt * 100) / 100))  # Animation title
    tk.update()  # Update animation frame
    time.sleep(0.001)  # Wait between loops

Tk.mainloop(canvas)  # Release animation handle (close window to finish)