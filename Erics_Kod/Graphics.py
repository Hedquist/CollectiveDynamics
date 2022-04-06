import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import time

res = 700  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3))) # Set height x width window
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res) # Place canvas with origin in x och y
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Raycasting

# Parameters of the agents
R_interaction = 20 # Interaction radius
R = 4 # Radius of agent
eta = 0.1 # Diffusional noise constant
FOV_angle = np.pi/3 # Field of view angle
half_FOV = FOV_angle/2
casted_rays = 5
step_angle = FOV_angle/(casted_rays-1)

# Parameters for the system
R_obstacle = 5
width_rectangular = 10
height_rectangular = 10

# Physical parameters of the system
t = 100000  # Simulation time
dt = 0.03  # Time step
l = 100 # Size of box
V = 20  # Particle velocity
N = 2 # Number of particles

x_agent=np.array([0, l / 2])
y_agent=np.array([0, l / 2])
phi = np.array([3*np.pi/2,0])
x_circular_obstacles = [0.5*2*l-l, 0.5*2*l-l]
y_circular_obstacles = [0.75*2*l-l, 0.25*2*l-l]
x_rectangular_obstacles = [0.25*2*l-l, 0.75*2*l-l]
y_rectuangular_obstacles = [0.5*2*l-l, 0.5*2*l-l]

agents = []
listR_f = []
list_of_arcs = []
list_of_rays = []
list_of_rays_coordinates = [[] for i in range(N)]
circular_obstacles = []
rectangular_obstacles = []


'''
x = np.random.rand(n) * 2 * l - l  # x coordinates # Generate numbers in interval [0,1)
y = np.random.rand(n) * 2 * l - l  # y coordinates
phi = np.random.rand(N) * 2 * np.pi  # orientations                # Initialization
particle_coordinates = np.array(list(zip(x,y)))

'''

def draw_particles():

    for j in range(N):  # Generate animated particles in Canvas
        # Convert to canvas coordinates
        agents.append(canvas.create_oval((x_agent[j] - R + l) * res / l / 2,
                                         (y_agent[j] - R + l) * res / l / 2,
                                         (x_agent[j] + R + l) * res / l / 2,
                                         (y_agent[j] + R + l) * res / l / 2,
                                         outline=ccolor[0], fill=ccolor[0])) # x0,y0 - x1,y1
        '''listR_f.append(canvas.create_oval((x_agent[j] - R_interaction + l) * res / l / 2, 
                                          (y_agent[j] - R_interaction + l) * res / l / 2, 
                                          (x_agent[j] + R_interaction + l) * res / l / 2,
                                          (y_agent[j] + R_interaction + l) * res / l / 2, 
                                          outline=ccolor[0])) # x0,y0 - x1,y1 '''
def cast_rays():

    for j in range(N):
        start_angle = phi[j] - half_FOV # Startvinkel
        start_angle_arc = start_angle # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            list_of_rays.append(canvas.create_line((x_agent[j] + l) * res / l / 2,
                                                      (y_agent[j] + l) * res / l / 2,
                                                      (x_agent[j] + R_interaction * np.cos(start_angle) + l) * res / l / 2,
                                                      (y_agent[j] + R_interaction * np.sin(start_angle) + l) * res / l / 2, fill=ccolor[2]))
            list_of_rays_coordinates[j].append([x_agent[j] + R_interaction * np.cos(start_angle), y_agent[j] + R_interaction * np.sin(start_angle)])
            start_angle += step_angle # Uppdaterar vinkel för ray

        list_of_arcs.append(canvas.create_arc((x_agent[j] - R_interaction + l) * res / l / 2,
                                               (y_agent[j] - R_interaction + l) * res / l / 2,
                                               (x_agent[j] + R_interaction + l) * res / l / 2,
                                              (y_agent[j] + R_interaction + l ) * res / l / 2,
                                              start=np.rad2deg(2*np.pi-phi[j]-half_FOV),extent=np.rad2deg(FOV_angle)
                                              ,style=ARC, fill=ccolor[2], outline=ccolor[2])) # Vinkel räknas motsols

def draw_circular_obstacles():

    for j in range(N):
        circular_obstacles.append(canvas.create_oval((x_circular_obstacles[j] - R_obstacle + l) * res / l / 2,
                                                      (y_circular_obstacles[j] - R_obstacle + l) * res / l / 2,
                                                      (x_circular_obstacles[j] + R_obstacle + l) * res / l / 2,
                                                      (y_circular_obstacles[j] + R_obstacle + l) * res / l / 2,
                                                      outline=ccolor[5], fill=ccolor[3]))  # x0,y0 - x1,y1))

def draw_rectangular_obstacles():
    for j in range(N):
        rectangular_obstacles.append(canvas.create_rectangle((x_rectangular_obstacles[j]  + l) * res / l / 2,
                                                              (y_rectuangular_obstacles[j]  + l) * res / l / 2,
                                                              (x_rectangular_obstacles[j] + width_rectangular + l) * res / l / 2,
                                                              (y_rectuangular_obstacles[j] + height_rectangular+ l) * res / l / 2,
                                                              outline=ccolor[5],fill=ccolor[4])) # x0,y0 - x1,y1))))
def proj_of_u_on_v(u,v):
    v_norm = np.sqrt(sum(v**2))
    return (np.dot(u, v)/v_norm**2)*v


def distance(r1,r2):
    return np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)


draw_particles()
cast_rays()
draw_circular_obstacles()
draw_rectangular_obstacles()




Tk.mainloop(canvas)  # Release animation handle (close window to finish)