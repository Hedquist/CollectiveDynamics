import cdist as cdist
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from scipy.spatial.distance import *
import time
from itertools import chain

res = 700  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))  # Set height x width window
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)  # Place canvas with origin in x och y
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

# Parameters of the fishes
fish_interaction_radius = 10  # Interaction radius
fish_graphic_radius = 4  # Radius of agent
fish_noise = 0.1  # Diffusional noise constant
fish_arrow_length = fish_interaction_radius / 2
fish_obst_detect_radius = fish_graphic_radius + 3
debug_visual = False

# Parameters for shark

# Physical parameters of the system
simulation_iterations = 100000  # Simulation time
time_step = 0.03  # Time step
canvas_length = 100  # Size of box
fish_count = 1  # Number of particles
fish_speed = 20

x = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
y = np.array(np.random.rand(fish_count) * 2 * canvas_length - canvas_length)
fish_coords = np.column_stack((x, y))
fish_orientations = np.random.rand(fish_count) * 2 * np.pi  # orientations

# if debug_visual:
#     x = np.array(np.ones(fish_count) * -5)
#     y = np.array(np.ones(fish_count) * 30)
#     fish_coords = np.column_stack((x, y))
#     fish_orientations = np.ones(fish_count) * 3/2 * np.pi  # orientations

circ_obst_coords = []
rect_obst_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []

with open('Obstacles3', 'r') as filestream:
    next(filestream)  # Skip first row
    for line in filestream:  # Read every row
        if line != "\n":
            currentline = line.split(',')
            if ('None' not in currentline[:3]):
                circ_obst_coords.append([float(currentline[0]), float(currentline[1])])
                circ_obst_radius.append(float(currentline[2]))
            if ('None' not in currentline[3:]):
                rect_obst_coords.append([float(currentline[3]), float(currentline[4])])
                rect_obst_width.append(float(currentline[5]))
                rect_obst_height.append(float(currentline[6]))
    circ_obst_coords, rect_obst_coords = np.array(circ_obst_coords), np.array(rect_obst_coords)

circ_obst_radius = np.array(circ_obst_radius)
rect_obst_width = np.array(rect_obst_width)
rect_obst_height = np.array(rect_obst_height)


obst_type = ['rect', 'circ'] #
obst_coords = [rect_obst_coords, circ_obst_coords] #
rect_obst_corner_coords = []

# Diverse tomma listor, mest grafik
circ_and_rect_obst_coords = np.concatenate((circ_obst_coords, rect_obst_coords))
fish_canvas_graphics = []
fish_interaction_radius_canvas_graphics = []
fish_obst_detect_radius_canvas_graphics = []
circ_obst_canvas_graphics = []
rect_obst_canvas_graphics = []
fish_direction_arrow_graphics = []

# Ta fram hörnen till rektangulära hinder
def calculate_rectangle_corner_coordinates(position, base, height):
    x_c = position[0]
    y_c = position[1]
    h = float(height)
    b = float(base)

    A = [x_c - b, y_c - h]
    B = [x_c + b, y_c - h]
    C = [x_c - b, y_c + h]
    D = [x_c + b, y_c + h]

    return [A, B, C, D]


# Ta fram hörnen till rektangulära hinder och lägg det i en lista
for i in range(len(rect_obst_coords)):
    rect_obst_corner_coords.append(
        calculate_rectangle_corner_coordinates(rect_obst_coords[i], rect_obst_width[i], rect_obst_height[i]))


# Funktion för att fisken startposition inte hamnar i ett hinder
def generate_fish_not_inside_obstacle_coordinates():
    for j in range(fish_count):
        while (True):  # Check if the obstacle is within agent
            distance_to_obstacles = []
            for k in range(circ_and_rect_obst_coords.shape[0]):
                distance_to_obstacles.append(np.linalg.norm(fish_coords[j] - circ_and_rect_obst_coords[k]))
            if (np.min(distance_to_obstacles) < fish_interaction_radius):
                fish_coords[j][0] = np.random.rand() * 2 * canvas_length - canvas_length
                fish_coords[j][1] = np.random.rand() * 2 * canvas_length - canvas_length
            else:
                break


# Ritar ut fiskar och dess interaktionsradie
def draw_fishes():
    for j in range(fish_count):  # Generate animated particles in Canvas
        # Convert to canvas coordinates
        fish_canvas_graphics.append(
            canvas.create_oval((fish_coords[j][0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               (fish_coords[j][1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[0], fill=ccolor[0]))
        if debug_visual:
            fish_interaction_radius_canvas_graphics.append(
                canvas.create_oval((fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                                   outline=ccolor[2], width=1))
            fish_obst_detect_radius_canvas_graphics.append(
                canvas.create_oval((fish_coords[j][0] - fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] - fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][0] + fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                                   (fish_coords[j][1] + fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                                   outline=ccolor[2], width=1))
        fish_direction_arrow_graphics.append(canvas.create_line((fish_coords[j][0] + fish_graphic_radius * np.cos(
            fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j][1] + fish_graphic_radius * np.sin(
                                                                    fish_orientations[
                                                                        j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j][0] + (
                                                                        fish_graphic_radius + fish_arrow_length) * np.cos(
                                                                    fish_orientations[
                                                                        j]) + canvas_length) * res / canvas_length / 2,
                                                                (fish_coords[j][1] + (
                                                                        fish_graphic_radius + fish_arrow_length) * np.sin(
                                                                    fish_orientations[
                                                                        j]) + canvas_length) * res / canvas_length / 2,
                                                                arrow=LAST))


# Ritar cirkulära hinder
def draw_circular_obstacles():
    for j in range(circ_obst_coords.shape[0]):
        circ_obst_canvas_graphics.append(
            canvas.create_oval((circ_obst_coords[j][0] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j][1] - circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j][0] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               (circ_obst_coords[j][1] + circ_obst_radius[j] + canvas_length) * res / canvas_length / 2,
                               outline=ccolor[5], fill=ccolor[3]))


# Ritar rektangulära hinder
def draw_rectangular_obstacles():
    for j in range(rect_obst_coords.shape[0]):
        rect_obst_canvas_graphics.append(canvas.create_rectangle(
            (rect_obst_coords[j][0] + rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j][1] + rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j][0] - rect_obst_width[j] + canvas_length) * res / canvas_length / 2,
            (rect_obst_coords[j][1] - rect_obst_height[j] + canvas_length) * res / canvas_length / 2,
            outline=ccolor[5], fill=ccolor[4]))


# Beräknar arean av en triangel givet tre punkter
def triangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) +
                x2 * (y3 - y1) +
                x3 * (y1 - y2)) / 2.0)


# Kollar om en punkt är innanför eller utanför en given rektangel med fyra hörn
def is_point_outside_rectangle(rectangle_corner_coords, point, outside):
    x = point[0]
    y = point[1]

    x1 = rectangle_corner_coords[0][0]
    y1 = rectangle_corner_coords[0][1]
    x2 = rectangle_corner_coords[1][0]
    y2 = rectangle_corner_coords[1][1]
    x3 = rectangle_corner_coords[2][0]
    y3 = rectangle_corner_coords[2][1]
    x4 = rectangle_corner_coords[3][0]
    y4 = rectangle_corner_coords[3][1]

    A = (triangle_area(x1, y1, x2, y2, x3, y3) +  # Calculate area of rectangel
         triangle_area(x1, y1, x4, y4, x3, y3))
    A1 = triangle_area(x, y, x1, y1, x2, y2)  # Calculate area of triangle PAB
    A2 = triangle_area(x, y, x2, y2, x3, y3)  # Calculate area of triangle PBC
    A3 = triangle_area(x, y, x3, y3, x4, y4)  # Calculate area of triangle PCD
    A4 = triangle_area(x, y, x1, y1, x4, y4)  # Calculate area of triangle PAD

    # Check if sum of A1, A2, A3 and A4 is same as A
    # 0.01 Felmarginal då datorn inte räknar exakt
    if (outside):
        return (A1 + A2 + A3 + A4 - 0.1 > A)
    else:
        return (A1 + A2 + A3 + A4 < A)



def calculate_distance(coords, coord):  # Räknar ut avstånd mellan punkterna coords och punkten coord
    return np.minimum(
        np.sqrt(((coords[:, 0]) % (2 * canvas_length) - (coord[0]) % (2 * canvas_length)) ** 2 + (
                (coords[:, 1]) % (2 * canvas_length) - (coord[1]) % (2 * canvas_length)) ** 2),
        np.sqrt((coords[:, 0] - coord[0]) ** 2 + (coords[:, 1] - coord[1]) ** 2))

def update_position(coords, speed, orientations):  # Uppdaterar en partikels position
    coords[:, 0] = (coords[:, 0] + speed * np.cos(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    coords[:, 1] = (coords[:, 1] + speed * np.sin(orientations) * time_step + canvas_length) % (
            2 * canvas_length) - canvas_length
    return coords

# Kallar på de grafiska funktionerna
#generate_fish_not_inside_obstacle_coordinates()
draw_fishes()
draw_circular_obstacles()
draw_rectangular_obstacles()

for t in range(simulation_iterations):
    fish_coords = update_position(fish_coords, fish_speed, fish_orientations)  # Uppdatera fiskposition
    for j in range(len(fish_coords)):  # Updating animation coordinates fisk
        canvas.coords(fish_canvas_graphics[j],
                      (fish_coords[j, 0] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 1] - fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 0] + fish_graphic_radius + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j, 1] + fish_graphic_radius + canvas_length) * res / canvas_length / 2)
        if debug_visual:
            canvas.coords(fish_interaction_radius_canvas_graphics[j],
                          (fish_coords[j][0] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][1] - fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][0] + fish_interaction_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][
                               1] + fish_interaction_radius + canvas_length) * res / canvas_length / 2)
            canvas.coords(fish_obst_detect_radius_canvas_graphics[j],
                          (fish_coords[j][0] - fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][1] - fish_obst_detect_radius + canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][0] + fish_obst_detect_radius+ canvas_length) * res / canvas_length / 2,
                          (fish_coords[j][1] + fish_obst_detect_radius + canvas_length) * res / canvas_length / 2)
        canvas.coords(fish_direction_arrow_graphics[j],
                      (fish_coords[j][0] + fish_graphic_radius * np.cos(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][1] + fish_graphic_radius * np.sin(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][0] + (fish_graphic_radius + fish_arrow_length) * np.cos(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2,
                      (fish_coords[j][1] + (fish_graphic_radius + fish_arrow_length) * np.sin(
                          fish_orientations[j]) + canvas_length) * res / canvas_length / 2)

        # Overlapp circular obstacles
        # circ_obst_overlap_distances = calculate_distance(circ_obst_coords, fish_coords[j])
        # angles = np.arctan2(circ_obst_coords[:,1] - fish_coords[j,1], circ_obst_coords[:,0] - fish_coords[j,0])  # Directions of others array from the particle
        # overlapp = circ_obst_overlap_distances < (fish_graphic_radius + circ_obst_radius)  # Applying
        # for ind in np.where(overlapp)[0]:
        #     fish_coords[j,0] = fish_coords[j,0] + (circ_obst_overlap_distances[ind] - (fish_graphic_radius + circ_obst_radius[ind])) * np.cos(angles[ind]) / 2
        #     fish_coords[j,1] = fish_coords[j,1] + (circ_obst_overlap_distances[ind] - (fish_graphic_radius + circ_obst_radius[ind])) * np.sin(angles[ind]) / 2


        # # Overlapp fishes
        fish_overlap_distances = calculate_distance(fish_coords, fish_coords[j])
        angles = np.arctan2(fish_coords[:,1] - fish_coords[j,1], fish_coords[:,0] - fish_coords[j,0])  # Directions of others array from the particle
        overlapp = fish_overlap_distances < (2 * fish_graphic_radius)  # Applying
        overlapp[j] = False  # area extraction
        for ind in np.where(overlapp)[0]:
            fish_coords[j,0] = fish_coords[j,0] + (fish_overlap_distances[ind] - 2 * fish_graphic_radius) * np.cos(angles[ind]) / 2
            fish_coords[j,1] = fish_coords[j,1] + (fish_overlap_distances[ind] - 2 * fish_graphic_radius) * np.sin(angles[ind]) / 2
            fish_coords[ind] = fish_coords[ind] - (fish_overlap_distances[ind] - 2 * fish_graphic_radius) * np.cos(angles[ind]) / 2
            fish_coords[ind] = fish_coords[ind] - (fish_overlap_distances[ind] - 2 * fish_graphic_radius) * np.sin(angles[ind]) / 2

        # Circular obstacle avoidance
        circ_obst_avoid_angle = 0
        fish_circ_obst_distances = calculate_distance(circ_obst_coords, fish_coords[j])
        detect_circ_obst = fish_circ_obst_distances < fish_obst_detect_radius +  circ_obst_radius

        if True in detect_circ_obst: # Om detekterar hinder

            ind = np.where(detect_circ_obst)[0] # Tar index på de detekterade hindren
            #time.sleep(0.5)
            closest_circ_obst_index = ind[np.argmin(calculate_distance(circ_obst_coords[ind],fish_coords[j])-circ_obst_radius[ind])] # Index för den närmsta objektet
            circ_obst_angle = np.arctan2(circ_obst_coords[closest_circ_obst_index,1],circ_obst_coords[closest_circ_obst_index,0]) # Räknar vinklar på dessa
            circ_obst_dist = calculate_distance(np.array([circ_obst_coords[closest_circ_obst_index]]),fish_coords[j])-circ_obst_radius[closest_circ_obst_index]
            center_to_fish = circ_obst_coords[closest_circ_obst_index] - fish_coords[j]
            center_angle = np.arctan2(center_to_fish[1],center_to_fish[0])
            positive_angle_center = 2*np.pi + center_angle if center_angle < 0 else center_angle
            positive_angle_fish = 2*np.pi + fish_orientations[j] if fish_orientations[j] < 0 else fish_orientations[j]
            # print(np.rad2deg(positive_angle_center))
            # print(np.rad2deg(positive_angle_fish))
            direction = np.sign(positive_angle_center-positive_angle_fish)
            #print(np.rad2deg(circ_obst_angle))
            #print(np.rad2deg(fish_orientations[j]))
            force = 1/circ_obst_dist
            circ_obst_avoid_angle = force*direction*np.pi/4
            print(np.rad2deg(circ_obst_avoid_angle[0]))



        inter_fish_distances = calculate_distance(fish_coords, fish_coords[
            j])  # Räknar ut avstånd mellan fisk j och alla andra fiskar
        fish_in_interaction_radius = inter_fish_distances < fish_interaction_radius  # Vilka fiskar är inom en fisks interraktionsradie
        fish_orientations[j] =  np.angle(np.sum(np.exp(fish_orientations[fish_in_interaction_radius] * 1j))) \
                                +  fish_noise * np.random.uniform(-1 / 2,1 / 2) + circ_obst_avoid_angle

    tk.title('Iteration =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
Tk.mainloop(canvas)  # Release animation handle (close window to finish)
