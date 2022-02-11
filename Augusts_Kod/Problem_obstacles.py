import numpy as np
from tkinter import *
from scipy.spatial import *
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.interpolate import interp1d
import time
from shapely.geometry import Polygon

# Skapa canvas att rita på
res = 500  # Resolution of the animation
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#17888E', '#C1D02B', '#9E00C9', '#D80000', '#E87B00', '#9F68D3', '#4B934F']

N = 10
l = 10

Rp = 0.5  # Fysisk, utritad radie
Rpf = 1  # Interaktionsradies för prey-agenter
vp = 0.1  # Prey-agenternas hastighet
RObs = 3  # Fysisk, utritad radie för obstacles
RObsf = 4  # Interaktionsradie för obstacles

dt = 1
eta = 0.01
iter = 10000

# Slumpa ut initialvillkor
r = np.array(np.random.rand(N, 2))
r = np.array(r) * 2 * l - l
phi = np.random.rand(N) * 2 * np.pi
for i in range(N // 2):  # För att få fördela initialpositioner även på negativa axlar
    r[i] = -r[i]

# Specifierade initialvillkor för hindren
rObs = np.array([[-5, -5], [5, 5]])

particles = []
obstacles = []
text = []


# Beräknar cluster-coefficienten för N partiklar med positioner r och interaktionsradie Rf
def cluster_coeff(r, Rf, N):
    v = Voronoi(r)
    area = np.zeros(v.npoints)
    coeff = 0
    for i, reg_num in enumerate(v.point_region):
        # clock = time.time()
        indices = v.regions[reg_num]

        if -1 not in indices:  # some regions can be opened
            area = Polygon(v.vertices[indices]).area
            if area < Rf ** 2 * np.pi:
                coeff = coeff + 1
        # print('Update position time: {t:10f}'.format(t=time.time() - clock))

    return coeff / N


# Lägg till alla partiklar och text på canvas:en
for i in range(rObs.size // 2):
    # also: make sure that following agents are not added inside the obstacles
    obstacles.append(canvas.create_oval((rObs[i, 0] - RObs + l) * res / l / 2,
                                        (rObs[i, 1] - RObs + l) * res / l / 2,
                                        (rObs[i, 0] + RObs + l) * res / l / 2,
                                        (rObs[i, 1] + RObs + l) * res / l / 2,
                                        outline=ccolor[3], fill=ccolor[3]))

for j in range(N):
    particles.append(canvas.create_oval((r[j, 0] - Rp + l) * res / l / 2,
                                        (r[j, 1] - Rp + l) * res / l / 2,
                                        (r[j, 0] + Rp + l) * res / l / 2,
                                        (r[j, 1] + Rp + l) * res / l / 2,
                                        outline=ccolor[0], fill=ccolor[0]))

GA = canvas.create_text(100, 20, text=1 / N * np.linalg.norm([np.sum(np.cos(phi)), np.sum(np.sin(phi))]))
GC = canvas.create_text(100, 40, text=cluster_coeff(r, Rpf, N))

for t in range(iter):  # Main loop där agenterna kontinuerligt uppdateras

    # Uppdatera alla agenter position
    r[:, 0] = (r[:, 0] + vp * np.cos(phi) * dt + l) % (2 * l) - l
    r[:, 1] = (r[:, 1] + vp * np.sin(phi) * dt + l) % (2 * l) - l

    for j in range(N):  # Alla agenter loopas igenom för att beräkna hur deras riktning ska uppdateras
        canvas.coords(particles[j],
                      (r[j, 0] - Rp + l) * res / l / 2,
                      (r[j, 1] - Rp + l) * res / l / 2,
                      (r[j, 0] + Rp + l) * res / l / 2,
                      (r[j, 1] + Rp + l) * res / l / 2, )  # Updating animation coordinates

        interact = np.array([j])
        dist = np.minimum(np.sqrt(
            ((r[:, 0]) % (2 * l) - (r[j, 0]) % (2 * l)) ** 2 + ((r[:, 1]) % (2 * l) - (r[j, 1]) % (2 * l)) ** 2),
            np.sqrt((r[:, 0] - r[j, 0]) ** 2 + (
                    r[:, 1] - r[j, 1]) ** 2))  # Beräkna avståndet från j:te agenten till alla andra
        interact = dist < Rpf  # Index för de agenter som är inom denna agents radie

        resultAngle = 0
        # Beräknar avståndet från j:te agenten till alla obstacles
        obsDist = np.minimum(np.sqrt(
            ((rObs[:, 0]) % (2 * l) - (r[j, 0]) % (2 * l)) ** 2 + ((rObs[:, 1]) % (2 * l) - (r[j, 1]) % (2 * l)) ** 2),
            np.sqrt((rObs[:, 0] - r[j, 0]) ** 2 + (rObs[:, 1] - r[j, 1]) ** 2))
        for i in range(2):
            if obsDist[i] < Rpf + RObsf:  # Om i:te hindret är inom radien för agenten
                # Beroende på om det minsta avståndet är inom rutan eller periodiskt blir det en
                # annan vektor mellan agent och hinder (collVect)
                if np.sqrt(
                        ((rObs[i, 0]) % (2 * l) - (r[j, 0]) % (2 * l)) ** 2 + (
                                (rObs[i, 1]) % (2 * l) - (r[j, 1]) % (2 * l)) ** 2) < np.sqrt(
                    (rObs[i, 0] - r[j, 0]) ** 2 + (rObs[i, 1] - r[j, 1]) ** 2):
                    collVect = np.array([((rObs[i, 0]) % (2 * l) - (r[j, 0]) % (2 * l)), (
                            (rObs[i, 1]) % (2 * l) - (r[j, 1]) % (2 * l))])
                else:
                    collVect = np.array([(rObs[i, 0] - r[j, 0]), (rObs[i, 1] - r[j, 1])])

                freeAngle = np.arctan(RObsf + Rpf / obsDist[
                    i])  # Vinkeln sett från mitten av obstacle till dess fulla radie + agentens radie
                # Vinkeln mellan agentens riktning och
                # vektorn som går från agenten till hindrets centrum
                angle = np.arctan2(collVect[1], collVect[0]) - phi[j]
                # Hur stor vinkel som ska adderas till agentens riktning för att undvika hindret,
                # desto närmare hindret desto större vinkel
                resultAngle = np.pi / 2 * (1 - obsDist[i] / (Rpf + RObsf))
                angle = np.arcsin(np.sin(angle)) # Detta är endast för att omvandla från 0, 2*pi till -pi, pi
                if np.abs(angle) < freeAngle: # Om den faktiskt behöver avvika överhuvudtaget
                    if angle > 0: # Så att den väljer att vika av åt det håll den redan var påväg
                        resultAngle = 0 - resultAngle
                else:
                    resultAngle = 0

        # Beräkna uppdaterade vinkeln
        phi[j] = np.angle(np.sum(np.exp(phi[interact] * 1j))) + eta * np.random.uniform(-1 / 2,
                                                                                        1 / 2) * dt + resultAngle

    # Beräkna och sen skriv ut dess konstanter
    globalAl = 1 / N * np.linalg.norm([np.sum(np.cos(phi)), np.sum(np.sin(phi))])
    clustCoeff = cluster_coeff(r, Rpf, N)

    canvas.itemconfig(GA, text='Global Alignment: {:.3f}'.format(globalAl))
    canvas.itemconfig(GC, text='Global Clustering: {:.3f}'.format(clustCoeff))

    tk.title('t =' + str(t))
    tk.update()  # Update animation frame
    time.sleep(0.01)  # Wait between loops
tk.mainloop()
