import numpy as np

canvas_length = 100
circ_radius = 5
R = circ_radius

circ_coord = np.array([90,90])
rect_corners_coords = np.array([[-95,85],[-85,85],[-85,95],[-95,95]])
Xc, Yc = circ_coord[0], circ_coord[1]  # Fiskens koordinater

X1, Y1 = rect_corners_coords[0,0], rect_corners_coords[0,1] # Ena hörnet
X2, Y2 = rect_corners_coords[3,0],rect_corners_coords[3,1] # Andra hörnet

NearestX = np.maximum(X1, np.minimum(Xc, X2)) # Tar fram de närmsta punkten
NearestY = np.maximum(Y1, np.minimum(Yc, Y2))

print(NearestX, NearestY, 'Nearest')
Dx = Xc - NearestX # Avståndet från närmsta punkten på rektangeln till cirkelns centrum
Dy = Yc - NearestY
# Avstånd eller boolean om cirkeln är innanför rektangeln
print(Dx,Dy,'DD')

Dx_modulo = (NearestX) % (2 * canvas_length) - (circ_coord[0]) % (2 * canvas_length)
Dy_modulo = (NearestY) % (2 * canvas_length) - (circ_coord[1]) % (2 * canvas_length)
print(Dx_modulo, Dy_modulo, 'DD_modulo')

Min = min(np.sqrt(Dx**2 + Dy**2), np.sqrt(Dx_modulo**2 + Dy_modulo**2))
print(Min, 'min')

x1 = [0,2]
x2 = [1,1]
print(np.minimum(x1,x2))