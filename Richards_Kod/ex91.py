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
plt.figure(1)
figv, axsv=plt.subplots(len(v), sharex=True,sharey=True)
figv.suptitle('Variera fart [v]')
posx=[]
posy=[]


for j in range(len(v)):
    for i in range(500):
        phi = phi+np.sqrt(2 * Dr[1]) * np.random.randn()*np.sqrt(dt)
        x=x+v[j]*cos(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)
        y=y+v[j]*sin(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)

        posx.append(x)
        posy.append(y)


    axsv[j].plot(posx, posy)
    axsv[j].set_title('v='+str(v[j]))
    x=0
    y=0
    posx = []
    posy = []




plt.figure(2)
figDt, axsDt=plt.subplots(len(Dr), sharex=True,sharey=True)
figDt.suptitle('Variera Dt')
for j in range(len(Dr)):
    for i in range(500):
        phi = phi+np.sqrt(2 * Dr[1]) * np.random.randn()*np.sqrt(dt)
        x=x+v[3]*cos(phi)*dt+np.sqrt(2*Dt[j])*np.random.randn()*np.sqrt(dt)
        y=y+v[3]*sin(phi)*dt+np.sqrt(2*Dt[j])*np.random.randn()*np.sqrt(dt)

        posx.append(x)
        posy.append(y)


    axsDt[j].plot(posx, posy)
    axsDt[j].set_title('Dt='+str(Dt[j]))
    x=0
    y=0
    posx = []
    posy = []

plt.figure(3)
figDr, axsDr=plt.subplots(len(Dr), sharex=True,sharey=True)
figDr.suptitle('Variera Dr')
for j in range(len(Dr)):
    for i in range(500):
        phi = phi+np.sqrt(2 * Dr[j]) * np.random.randn()*np.sqrt(dt)
        x=x+v[3]*cos(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)
        y=y+v[3]*sin(phi)*dt+np.sqrt(2*Dt[1])*np.random.randn()*np.sqrt(dt)

        posx.append(x)
        posy.append(y)


    axsDr[j].plot(posx, posy)
    axsDr[j].set_title('Dr='+str(Dr[j]))
    x=0
    y=0
    posx = []
    posy = []



plt.show()
