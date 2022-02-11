import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt


n=1
dt=0.01
v=np.array([0,1,2,3])*1e-6
Dt=np.array([0,2,4,6])*1e-13
Dr=[0,0.5,1,2]


x=0
y=0
phi=0
plt.figure(0)
figv=plt.figure()
axsv=figv.add_subplot(111)

posx=[]
posy=[]


for j in range(len(v)):
    sd = np.array([])
    MSD = np.array([])
    for k in range(100):
        time = np.array([])

        for i in range(500):
            phi = phi+np.sqrt(2 * Dr[1]) * np.random.randn()*np.sqrt(dt)
            x=x+v[j]*cos(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)
            y=y+v[j]*sin(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)

            posx.append(x)
            posy.append(y)
            sd=np.append(sd, x ** 2 + y ** 2)

            time=np.append(time, dt*i)

    sd=np.reshape(sd,(500,100))
    print(sd.shape)

    MSD=sd.sum(1)/100
    print(MSD.shape)
    axsv.plot(time, MSD)
    axsv.set_title('v='+str(v[j]))
    axsv.set_yscale('log')
    axsv.set_xscale('log')
    x=0
    y=0
    posx = []
    posy = []






plt.show()