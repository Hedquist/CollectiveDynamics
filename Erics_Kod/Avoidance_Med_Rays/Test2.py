import numpy as np

circ_obst_coords = []
rect_obst_coords = []
rect_obst_corner_coords = []

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
rect_obst_corner_coords = np.array(rect_obst_corner_coords)

j = 0
fish_graphic_radius = 3  # Radius of agent
fish_coords = np.array([[10,10]])

Xc, Yc = fish_coords[j,0], fish_coords[j,1]  # Fiskens koordinater
X1, Y1 = rect_obst_corner_coords[:,0,0], rect_obst_corner_coords[:,0,1] # Ena hörnet
X2, Y2 = rect_obst_corner_coords[:,3,0],rect_obst_corner_coords[:,3,1] # Andra hörnet

Xn = np.maximum(X1, np.minimum(Xc,X2))
Yn = np.maximum(Y1, np.minimum(Yc,Y2))

Dx = Xn - Xc
Dy = Yn - Yc
fish_inside_rect_obst = (Dx * Dx + Dy * Dy) <= fish_graphic_radius * fish_graphic_radius


