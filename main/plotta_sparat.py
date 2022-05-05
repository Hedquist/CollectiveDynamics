import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('1till32.npy')
data2 = np.load('inget-beteende-1till32.npy')

fig, ax = plt.subplots(figsize=(7.5,6.5))
markers, caps, bars = ax.errorbar(data1[2,:], data1[0,:]/data1[2,:], yerr=data1[1,:]/data1[2,:], fmt='o', color='#2E86C1')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
markers, caps, bars = ax.errorbar(data2[2,:], data2[0,:]/data2[2,:], yerr=data2[1,:]/data2[2,:], fmt='o')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.xlabel('Antal rovdjur', fontsize=16)
plt.ylabel('Medelvärde av andel fångade bytesdjur [%]', fontsize=16)
ax.tick_params(axis='both', labelsize=16, right=True, top=True)
#ax.legend(('110% av bytens fart', '90% av bytens fart'), loc=2)


plt.show()
