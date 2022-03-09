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

# Parameters of the fishes
fish_interaction_radius = 10 # Interaction radius
fish_graphic_radius = 2 # Radius of agent
fish_noise = 0.1 # Diffusional noise constant
fish_arrow_length = fish_interaction_radius

# Raycasting main
step_angle = 2 * np.arctan(fish_graphic_radius / fish_interaction_radius)
casted_rays = 6
FOV_angle = step_angle * (casted_rays - 1)  # Field of view angle
half_FOV = FOV_angle / 2


# Parameters for shark


# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100 # Size of box
fish_speed = 20  # Particle velocity
fish_count = 1# Number of particles

x=np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
y=np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
fish_coords = np.column_stack((x, y))
fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations


circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []

with open('Obstacles', 'r') as filestream:
    next(filestream) # Skip first row
    for line in filestream: # Read every row
        if line is not "\n":
            currentline = line.split(',')
            if('None' not in currentline[:3]):
                circ_obst_coords.append( [float(currentline[0]), float(currentline[1])] )
                circ_obst_radius.append(float(currentline[2]))
            if('None'not in currentline[3:]):
                rect_obst_coords.append( [float(currentline[3]), float(currentline[4])] )
                rect_obst_width.append(float(currentline[5]))
                rect_obst_height.append(float(currentline[6]))
    circ_obst_coords, rect_obst_coords =  np.array(circ_obst_coords), np.array(rect_obst_coords)



obst_type = ['wall','rect','circ']
obst_coords = [ [[0,0]] , rect_obst_coords, circ_obst_coords ]
wall_corner_coords = np.array([[canvas_length,canvas_length],[-canvas_length,canvas_length],[-canvas_length,-canvas_length],[canvas_length,-canvas_length]])
rect_obst_corner_coords = []

# Diverse tomma listor, mest grafik
#circ_and_rect_obst_coords = np.concatenate((circ_obst_coords, rect_obst_coords))
fish_canvas_graphics = []
fish_interaction_radius_canvas_graphics = []
fish_canvas_rays_graphics = [[] for i in range(fish_count)]
rays_coords = [[] for i in range(fish_count)]
rays_angle_relative_velocity = [[] for i in range(fish_count)]
circ_obst_canvas_graphics = []
rect_obst_canvas_graphics = []
fish_direction_arrow_graphics = []

def draw_fishes():

    for j in range(fish_count):  # Generate animated particles in Canvas
        # Convert to canvas coordinates
        fish_canvas_graphics.append(canvas.create_oval((fish_coords[j][0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                                                       outline=ccolor[0], fill=ccolor[0])) # x0,y0 - x1,y1
        fish_interaction_radius_canvas_graphics.append(canvas.create_oval((fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                                                          (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                                                          (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                                                          (fish_coords[j][1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                                                          outline=ccolor[2],width=1)) # x0,y0 - x1,y1
        fish_direction_arrow_graphics.append(canvas.create_line( (fish_coords[j][0]  + fish_graphic_radius *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][1] + fish_graphic_radius *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][0] +  (fish_graphic_radius + fish_arrow_length ) *np.cos(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                       (fish_coords[j][1] + (fish_graphic_radius + fish_arrow_length ) *np.sin(fish_orientations[j]) + canvas_length) * res / canvas_length / 2,arrow=LAST ))# x0,y0 - x1,y1
# Ritar ut rays och lägger dess vinkel och spetsens koordinater i en lista
def cast_rays():

    for j in range(fish_count):
        start_angle = fish_orientations[j] - half_FOV # Startvinkel
        start_angle_arc = start_angle # Memorerar för j:te partikeln
        for ray in range(casted_rays):
            fish_canvas_rays_graphics[j].append(canvas.create_line((fish_coords[j][0] + canvas_length) * res / canvas_length / 2,
                                                                   (fish_coords[j][1] + canvas_length) * res / canvas_length / 2,
                                                                   (fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle) + canvas_length) * res / canvas_length / 2,
                                                                   (fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle) + canvas_length) * res / canvas_length / 2, fill=ccolor[3]))
            rays_coords[j].append([fish_coords[j][0] + fish_interaction_radius * np.cos(start_angle), fish_coords[j][1] + fish_interaction_radius * np.sin(start_angle)])
            rays_angle_relative_velocity[j].append(start_angle)
            start_angle += step_angle # Uppdaterar vinkel för ray

# Ritar cirkulära hinder
def draw_circular_obstacles():

    for j in range(circ_obst_coords.shape[0]):
        circ_obst_canvas_graphics.append(canvas.create_oval((circ_obst_coords[j][0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            (circ_obst_coords[j][1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                                            outline=ccolor[5], fill=ccolor[3]))  # x0,y0 - x1,y1))
# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    for j in range(rect_obst_coords.shape[0]):
        rect_obst_canvas_graphics.append(canvas.create_rectangle((rect_obst_coords[j][0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                                                                 (rect_obst_coords[j][1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                                                                 outline=ccolor[5], fill=ccolor[4])) # x0,y0 - x1,y1))))


draw_fishes()
cast_rays()
draw_circular_obstacles()
draw_rectangular_obstacles()




Tk.mainloop(canvas)  # Release animation handle (close window to finish)