import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('inget-beteende-1till32.npy')
data2 = np.load('slow_shark.npy')
fig, ax = plt.subplots(figsize=(7.5,6.5))
markers, caps, bars = ax.errorbar(data1[2,:], data1[0,:], yerr=data1[1,:], fmt='-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
'''markers, caps, bars = ax.errorbar(data2[2,:], data2[0,:], yerr=data2[1,:], fmt='-')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]'''
plt.xlabel('Tid', fontsize=16)
plt.ylabel('Medelvärde av andel fångade bytesdjur [%]', fontsize=16)
ax.tick_params(axis='both', labelsize=16, right=True, top=True)
ax.legend(('110% av bytens fart', '90% av bytens fart'), loc=2)


plt.show()
