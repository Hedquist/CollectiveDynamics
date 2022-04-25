import numpy as np

circ_obst_coords = []
rect_obst_coords = []
rect_obst_corner_coords = []

circ_obst_radius = []
rect_obst_width = []
rect_obst_height = []

with open('../Environments/Environment4', 'r') as filestream:
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

print()
circ_coord = np.array([10,5])
rect_coords = rect_obst_coords
circ_radius = 2
rect_width = rect_obst_width
rect_height = rect_obst_height

delta = circ_coord - rect_coords
print(delta,'delta')
min_index = [np.argmin(np.absolute(delta[i])) for i in range(len(delta)) ]# Tar fram närmaste x eller y koordinaten
print(min_index,'min_index')
displacement = [ [delta[index,min_index[index]] if min_index[index] == 0 else 0, delta[index,min_index[index]] if min_index[index] == 1 else 0 ] for index, element in enumerate(min_index)  ]
print(displacement,'displacement')
point_normal_coords = np.array(rect_coords + np.array(displacement)) # Position för normapunkten
print(point_normal_coords, 'point_normal_coords')
normal_vec = circ_coord - point_normal_coords # Normalens vektor
print(normal_vec, 'normal_vec')
normal_dist = np.array([np.linalg.norm(element) for element in normal_vec])
print(normal_dist, 'normal dist')
normal_angle = np.array(np.arctan2(normal_vec[:,1], normal_vec[:,0]))  # Directions of others array from the particle
print(normal_angle, 'normal angle')
actual_dist = normal_dist - (circ_radius + rect_width*np.cos(normal_angle) + rect_height*np.sin(normal_angle))
print(actual_dist)
dx,dy = np.absolute(normal_dist - (circ_radius + rect_width) ) * np.cos(normal_angle),np.absolute(normal_dist - (circ_radius + rect_height) ) * np.sin(normal_angle)
print(dx,dy)
dist = [ dx[index] if np.absolute(dx[index]) > np.absolute(dy[index]) else dy[index]  for index, element in enumerate(min_index)  ]
print(dist,'dist')

def distance_circ_to_rect(circ_coord,circ_radius, rect_coords,rect_width,rect_height):
    delta = circ_coord - rect_coords
    min_index_xy = [np.argmin(np.absolute(delta[i])) for i in range(len(delta))]# Tar fram närmaste x eller y koordinaten
    displacement = [[delta[index, min_index_xy[index]] if min_index_xy[index] == 0 else 0, delta[index, min_index_xy[index]] if min_index_xy[index] == 1 else 0] for index, element in enumerate(min_index_xy)]
    point_normal_coords = np.array(rect_coords + np.array(displacement)) # Position för normapunkten
    normal_vec = circ_coord - point_normal_coords # Normalens vektor
    normal_dist = np.array([np.linalg.norm(element) for element in normal_vec]) # Normal vektorns längd
    normal_angle = np.array(np.arctan2(normal_vec[:,1], normal_vec[:,0]))  # Vinkeln för normalvektorn, är 0 pi/2 3pi/2, 2pi
    actual_dist = normal_dist - (circ_radius + rect_width*np.cos(normal_angle) + rect_height*np.sin(normal_angle)) # Det avståndet som blir över
    return actual_dist

def distance_points_to_rect(point_coords, rect_coord, rect_width, rect_height):
    delta = point_coords - rect_coord
    min_index_xy = [np.argmin(np.absolute(delta[i])) for i in range(len(delta))]# Tar fram närmaste x eller y koordinaten
    displacement = [[delta[index, min_index_xy[index]] if min_index_xy[index] == 0 else 0, delta[index, min_index_xy[index]] if min_index_xy[index] == 1 else 0] for index, element in enumerate(min_index_xy)]
    point_normal_coords = np.array(rect_coord + np.array(displacement)) # Position för normapunkten
    normal_vec = point_coords - point_normal_coords # Normalens vektor
    normal_dist = np.array([np.linalg.norm(element) for element in normal_vec]) # Normal vektorns längd
    normal_angle = np.array(np.arctan2(normal_vec[:,1], normal_vec[:,0]))  # Vinkeln för normalvektorn, är 0 pi/2 3pi/2, 2pi
    actual_dist = np.absolute(normal_dist - (rect_width*np.cos(normal_angle) + rect_height*np.sin(normal_angle))) # Det avståndet som blir över
    return actual_dist

def distance_circ_to_rect_boolean(circ_coord, circ_radius, rect_corners_coords):
    R = circ_radius
    Xc, Yc = circ_coord[0], circ_coord[1]  # Fiskens koordinater
    X1, Y1 = rect_corners_coords[:,0,0], rect_corners_coords[:,0,1] # Ena hörnet
    X2, Y2 = rect_corners_coords[:,3,0],rect_corners_coords[:,3,1] # Andra hörnet

    NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
    NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

    Dx = NearestX - Xc # Avståndet från närmsta punkten på rektangeln till fiskens centrum
    Dy = NearestY - Yc
    circle_inside_rectangular = (Dx * Dx + Dy * Dy) <= R *R

    return circle_inside_rectangular

def distance_circ_to_rect(circ_coord, circ_radius, rect_corners_coords):
    R = circ_radius
    Xc, Yc = circ_coord[0], circ_coord[1]  # Fiskens koordinater
    X1, Y1 = rect_corners_coords[0,0], rect_corners_coords[0,1] # Ena hörnet
    X2, Y2 = rect_corners_coords[3,0],rect_corners_coords[3,1] # Andra hörnet

    NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
    NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

    Dx = NearestX - Xc # Avståndet från närmsta punkten på rektangeln till fiskens centrum
    Dy = NearestY - Yc

    dist = np.absolute(np.sqrt(Dx ** 2 + Dy ** 2)-R)
    return dist