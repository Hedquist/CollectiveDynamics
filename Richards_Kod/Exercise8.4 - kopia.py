import numpy as np
from tkinter import *
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import time
from shapely.geometry import Polygon

res = 500  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

N = 50
l = 100
R = 4
Rf = 10
v = 2
dt = 1
eta = 0.1

x = np.random.rand(N)*2*l - l    # x coordinates
y = np.random.rand(N)*2*l - l    # y coordinates
phi = np.random.rand(N)*2*np.pi  # orientations
XY=np.column_stack((x, y))

filename1='StartVal.npy'
filename2='initPhi.npy'
file1=np.save(filename1, XY)
file2=np.save(filename2, phi)

r = np.load('StartVal.npy')
phi = np.load('initPhi.npy')

for i in range(N // 2):
    r[i] = -r[i]

particles = []
text = []

N_shark=1
r_shark=np.column_stack((0.0, 0.0))

v_shark=1.8
phi_shark= np.random.rand(N_shark) * 2 * np.pi
shark = []

def update_position(r, v, phi, dt):
    r[:, 0] = (r[:, 0] + v * np.cos(phi) * dt + l) % (2 * l) - l
    print((v * np.cos(phi) * dt + l) % (2 * l) - l)
    r[:, 1] = (r[:, 1] + v * np.sin(phi) * dt + l) % (2 * l) - l
    return r

def distance(r1, r2, l):
    # print("X1: %5.3f, Y1: %5.3f, X2: %5.3f, Y2: %5.3f, deltaX: %5.3f, deltaY: %5.3f" % (r1[0], r1[1], r2[0], r2[1],
    # (r1[0]+l) % (2 * l) - (r2[0]+l) % (2 * l), (r1[1]+l) % (2 * l) - (r2[1]+l) % (2 * l)))
    return np.minimum(
        np.sqrt(((r1[0]) % (2 * l) - (r2[0]) % (2 * l)) ** 2 + ((r1[1]) % (2 * l) - (r2[1]) % (2 * l)) ** 2),
        np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2))


def cluster_coeff(r, Rf, N):

    v = Voronoi(r)
    area = np.zeros(v.npoints)
    coeff = 0
    for i, reg_num in enumerate(v.point_region):
        #clock = time.time()
        indices = v.regions[reg_num]

        if -1 not in indices:  # some regions can be opened
            area = Polygon(v.vertices[indices]).area
            if area < Rf ** 2 * np.pi:
                coeff = coeff + 1
        #print('Update position time: {t:10f}'.format(t=time.time() - clock))

    return coeff / N

for j in range(N_shark):
    shark.append(canvas.create_oval((r_shark[j, 0] - R + l) * res / l / 2,
                                        (r_shark[j, 1] - R + l) * res / l / 2,
                                        (r_shark[j, 0] + R + l) * res / l / 2,
                                        (r_shark[j, 1] + R + l) * res / l / 2,
                                        outline=ccolor[1], fill=ccolor[1]))
for j in range(N):
    particles.append(canvas.create_oval((r[j, 0] - R + l) * res / l / 2,
                                        (r[j, 1] - R + l) * res / l / 2,
                                        (r[j, 0] + R + l) * res / l / 2,
                                        (r[j, 1] + R + l) * res / l / 2,
                                        outline=ccolor[0], fill=ccolor[0]))
    '''
    closeParticles = 0
    for k in range(N):
        if k != j:
            if distance(r[j, :], r[k, :], l) < Rf:
                closeParticles = closeParticles + 1

    text.append(canvas.create_text(r[j, 0] + R, r[j, 1], text=closeParticles)) '''

GA = canvas.create_text(100, 20, text=1 / N * np.linalg.norm([np.sum(np.cos(phi)), np.sum(np.sin(phi))]))
GC = canvas.create_text(100, 40, text=cluster_coeff(r, Rf, N))

vor = Voronoi(r)
fig = voronoi_plot_2d(vor)
plt.gca().invert_yaxis()
#plt.show()

globalAl = np.zeros(4000)
clustCoeff = np.zeros(4000)

f = 0

for t in range(4000):
    r=update_position(r,v,phi,dt)
    r_shark=update_position(r_shark, v_shark, phi_shark, dt)
    sharkdists = np.minimum(np.sqrt(
        ((r[:, 0]) % (2 * l) - r_shark[0,0] % (2 * l)) ** 2 + ((r[:, 1]) % (2 * l) - r_shark[0,1] % (2 * l)) ** 2),
        np.sqrt((r[:, 0] - r_shark[0,0]) ** 2 + (r[:, 1] - r_shark[0,1]) ** 2))


    closest_fish = np.where(sharkdists == np.amin(sharkdists))[0] # Index of closest fish

    if np.sqrt(((r[closest_fish, 0]) % (2 * l) - r_shark[0,0] % (2 * l)) ** 2 + ((r[closest_fish, 1]) % (2 * l) - r_shark[0,1] % (2 * l)) ** 2)<np.sqrt((r[closest_fish, 0] - r_shark[0,0]) ** 2 + (r[closest_fish, 1] - r_shark[0,1]) ** 2):
        phi_shark = np.arctan2((r[closest_fish, 1]) % (2 * l) - r_shark[0, 1] % (2 * l), (r[closest_fish, 0]) % (2 * l) - r_shark[0, 0] % (2 * l))
    else:
        phi_shark = np.arctan2((r[closest_fish, 1] - r_shark[0, 1]), (r[closest_fish, 0] - r_shark[0, 0]))



    #sharkphi = np.arctan2(ydiff[yIndex] , xdiff[xIndex])
    for j in range(len(shark)):
        canvas.coords(shark[j],
                      (r_shark[j, 0] - R + l) * res / l / 2,
                      (r_shark[j, 1] - R + l) * res / l / 2,
                      (r_shark[j, 0] + R + l) * res / l / 2,
                      (r_shark[j, 1] + R + l) * res / l / 2, )
    for j in range(N):
        canvas.coords(particles[j],
                      (r[j, 0] - R + l) * res / l / 2,
                      (r[j, 1] - R + l) * res / l / 2,
                      (r[j, 0] + R + l) * res / l / 2,
                      (r[j, 1] + R + l) * res / l / 2, )  # Updating animation coordinates
        if j == closest_fish:
            canvas.itemconfig(particles[j], fill=ccolor[2])
        else:
            canvas.itemconfig(particles[j], fill=ccolor[0]) # Change color of fish closest to shark
        closeParticles = 0
        interact = np.array([j])
        sumTime = np.zeros(N)

        dist = np.minimum(np.sqrt(
            ((r[:, 0]) % (2 * l) - (r[j, 0]) % (2 * l)) ** 2 + ((r[:, 1]) % (2 * l) - (r[j, 1]) % (2 * l)) ** 2),
                          np.sqrt((r[:, 0] - r[j, 0]) ** 2 + (r[:, 1] - r[j, 1]) ** 2))
        interact = dist < Rf
        if sharkdists[j] < Rf:
            if np.sqrt(((r[j, 0]) % (2 * l) - r_shark[0,0] % (2 * l)) ** 2 + (
                    (r[j, 1]) % (2 * l) - r_shark[0,1] % (2 * l)) ** 2) < np.sqrt(
                    (r[j, 0] - r_shark[0,0]) ** 2 + (r[j, 1] - r_shark[0,1]) ** 2):
                phi[j] = np.arctan2((r[j, 1]) % (2 * l) - r_shark[0,1] % (2 * l),
                                      (r[j, 0]) % (2 * l) - r_shark[0,0] % (2 * l))
            else:
                phi[j] = np.arctan2((r[j, 1] - r_shark[0,1]), (r[j, 0] - r_shark[0,0]))
        else:
            phi[j] = np.angle(np.sum(np.exp(phi[interact] * 1j))) + eta * np.random.uniform(-1 / 2, 1 / 2)


        '''
        for k in range(N):

            if k != j:
                # print(distance(r[j, :], r[k, :], l))
                tid = time.time()
                if k < 10: #distance(r[j, :], r[k, :], l) < Rf:

                    closeParticles = closeParticles + 1
                    interact = np.append(interact, [k])

                sumTime[k] = (time.time() - tid)
            #print("sumTime: {:8f}".format(sumTime))
            #print("SpecificTime: {:10f}".format(time.time() - tid))
        print('Update position time: {t:10f}, sumTime: {a:10f}, particle: {c:1f}'.format(t=time.time() - clock, a=np.sum(sumTime), c=j))
        #print(sumTime)
        '''

        # print(closeParticles)
        #canvas.coords(text[j], (r[j, 0] + l) * res / l / 2, (r[j, 1] + l) * res / l / 2)
        #canvas.itemconfig(text[j], text=closeParticles)



    globalAl[t] = 1 / N * np.linalg.norm([np.sum(np.cos(phi)), np.sum(np.sin(phi))])
    clustCoeff[t] = cluster_coeff(r, Rf, N)

    canvas.itemconfig(GA, text='Global Alignment: {:.3f}'.format(globalAl[t]))
    canvas.itemconfig(GC, text='Global Clustering: {:.3f}'.format(clustCoeff[t]))
    '''
    pause = [10, 100, 500, 1000]
    if t in pause:
        print(pause[f])
        plt.plot(np.arange(pause[f]), globalAl[0:pause[f]])
        plt.plot(np.arange(pause[f]), clustCoeff[0:pause[f]])
        f = f + 1
        vor = Voronoi(r)
        fig = voronoi_plot_2d(vor)
        plt.gca().invert_yaxis()
        plt.show()
    '''

    tk.title('t =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
tk.mainloop()

plt.plot(np.arange(4000), globalAl)
plt.plot(np.arange(4000), clustCoeff)
plt.show()
