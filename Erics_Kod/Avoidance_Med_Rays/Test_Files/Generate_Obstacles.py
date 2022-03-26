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


# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100 # Size of box
fish_speed = 20  # Particle velocity
fish_count = 1# Number of particles

# Hinder info
circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []


obst_type = ['rect', 'circ']
rect_obst_corner_coords = []

circ_obst_canvas_graphics = []
rect_obst_canvas_graphics = []
box_canvas_graphics = []


# Ta fram hörnen till rektangulära hinder
def calculate_rectangle_corner_coordinates(position, base, height):
    x_c, y_c = position[0], position[1]
    b,h = float(base), float(height)

    A = [x_c - b, y_c - h]
    B = [x_c + b, y_c - h]
    C = [x_c - b, y_c + h]
    D = [x_c + b, y_c + h]

    return [A, B, C, D]


# Ta fram hörnen till rektangulära hinder och lägg det i en lista
for i in range(len(rect_obst_coords)):
    rect_obst_corner_coords.append(
        calculate_rectangle_corner_coordinates(rect_obst_coords[i], rect_obst_width[i], rect_obst_height[i]))
rect_obst_corner_coords = np.array(rect_obst_corner_coords)

# Ritar cirkulära hinder
def draw_circular_obstacles():
    if np.size(circ_obst_coords) != 0:
        for j in range(circ_obst_coords.shape[0]):
            circ_obst_canvas_graphics.append(
                canvas.create_oval((circ_obst_coords[j, 0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   (circ_obst_coords[j, 1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                                   outline=ccolor[5], fill=ccolor[3]))


# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    if np.size(rect_obst_coords) != 0:
        for j in range(rect_obst_coords.shape[0]):
            rect_obst_canvas_graphics.append(canvas.create_rectangle(
                (rect_obst_coords[j, 0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
                (rect_obst_coords[j, 1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
                outline=ccolor[5], fill=ccolor[4]))

    box_canvas_graphics.append(canvas.create_rectangle(5, 10,res-5,res-5,outline='black', fill=None, width=5))

def load_obstacles(obstacle_type, num_row, num_col, obstacle_size, displacement):
    horisontal_space = 2 * canvas_length/(num_col+1) # Mellanrum i horisentell led
    vertical_space = 2 * canvas_length/(num_row+1) # Mellanrum i vertikalled
    start_vertical = - canvas_length + vertical_space # Start i vertikalled, högst upp till vänster
    for i in range(num_row): # För varje rad
        start_horisontal = - canvas_length + 3/2*horisontal_space if displacement and i % 2 != 0 \
            else- canvas_length + 2*canvas_length/(num_col+1) # Förskjuts om True annars vanlig start vi horisontell led
        for j in range(num_col - 1 if displacement and i%2 != 0 else num_col): # För varje kolonn, minska antalet om displacement
            if obstacle_type == 'circles':
                circ_obst_coords.append([start_horisontal, start_vertical])
                circ_obst_radius.append(obstacle_size)
            elif obstacle_type == 'rectangles':
                rect_obst_coords.append([start_horisontal, start_vertical])
                rect_obst_width.append(obstacle_size)
                rect_obst_height.append(obstacle_size)
            start_horisontal += horisontal_space # Lägg till avståndet
        start_vertical += vertical_space # Gå till nästa rad

def load_circular_obstacles(num_row, num_col, obstacle_radius, displacement):
    horisontal_space = 2 * canvas_length/(num_col+1) # Mellanrum i horisentell led
    vertical_space = 2 * canvas_length/(num_row+1) # Mellanrum i vertikalled
    start_vertical = - canvas_length + vertical_space # Start i vertikalled, högst upp till vänster
    for i in range(num_row): # För varje rad
        start_horisontal = - canvas_length + 3/2*horisontal_space if displacement and i % 2 != 0 \
            else- canvas_length + 2*canvas_length/(num_col+1) # Förskjuts om True annars vanlig start vi horisontell led
        for j in range(num_col - 1 if displacement and i%2 != 0 else num_col): # För varje kolonn, minska antalet om displacement
            circ_obst_coords.append([start_horisontal, start_vertical])
            circ_obst_radius.append(obstacle_radius)
            start_horisontal += horisontal_space # Lägg till avståndet
        start_vertical += vertical_space # Gå till nästa rad

def load_rectangular_obstacles(num_row, num_col, width, height, displacement):
    horisontal_space = 2 * canvas_length/(num_col+1)
    vertical_space = 2 * canvas_length/(num_row+1)
    start_vertical = - canvas_length + vertical_space
    for i in range(num_row): # För varje rad
        start_horisontal = - canvas_length + 3/2*horisontal_space if displacement and i % 2 != 0 \
            else- canvas_length + 2*canvas_length/(num_col+1)
        for j in range(num_col - 1 if displacement and i%2 != 0 else num_col):
            rect_obst_coords.append([start_horisontal, start_vertical])
            rect_obst_width.append(width)
            rect_obst_height.append(height)
            start_horisontal += horisontal_space
        start_vertical += vertical_space

num_of_obstacles = 5
obstacle_size = 5
#load_circular_obstacles(num_of_obstacles,num_of_obstacles,5, True)
#load_rectangular_obstacles(num_of_obstacles,num_of_obstacles+1,5,5,True)
load_obstacles('rectangles', num_of_obstacles, num_of_obstacles, obstacle_size, True)

circ_obst_coords, rect_obst_coords = np.array(circ_obst_coords), np.array(rect_obst_coords)

circ_obst_radius = np.array(circ_obst_radius)
rect_obst_width = np.array(rect_obst_width)
rect_obst_height = np.array(rect_obst_height)

draw_circular_obstacles()
draw_rectangular_obstacles()
Tk.mainloop(canvas)  # Release animation handle (close window to finish)
