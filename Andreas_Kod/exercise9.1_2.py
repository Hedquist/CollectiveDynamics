import numpy as np
from matplotlib import pyplot as plt

# Konstanter
dt = 0.01
DT = [1e-14, 5e-14, 2e-13, 5e-13, 1e-12]
DR = [0.05, 0.1, 0.5, 1]
v = [0, 1e-6, 2e-6, 3e-6]
s = len(v)
N = 10000
num = 100

x = np.zeros(N)
y = np.zeros(N)
phi = np.ones(N)
t = np.ones(N-1)
for j in range(s):
    MSD = np.zeros(N - 1)
    for k in range(num):
        for i in range(N-1):
            W = np.random.randn()/np.sqrt(dt)
            x[i+1] = x[i] + (v[j] * np.cos(phi[i]) + np.sqrt(2 * DT[2]) * W) * dt
            W = np.random.randn()/np.sqrt(dt)
            y[i+1] = y[i] + (v[j] * np.sin(phi[i]) + np.sqrt(2 * DT[2]) * W) * dt
            W = np.random.randn()/np.sqrt(dt)
            phi[i+1] = phi[i] + np.sqrt(2*DR[2]) * W * dt

            MSD[i] += x[i+1]**2/num + y[i+1]**2/num
            t[i] = (i+1) * dt

    plt.figure(1)
    plt.subplot(1, s, j+1)
    plt.plot(x*1e6, y*1e6)
    plt.title(f"v = {v[j]*1e6} um/s")
    # plt.title(f"DT = {DT[j]}")
    # plt.title(f"DR = {DR[j]}")
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")

    plt.figure(2)
    plt.loglog(t, MSD)

plt.show()
