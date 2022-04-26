import numpy as np
import matplotlib.pyplot as plt

data = np.load('haj1till15.npy')
fig, ax = plt.subplots(figsize=(7.5,6.5))
markers, caps, bars = ax.errorbar(data[2,:], data[0,:], yerr=data[1,:], fmt='o', color='#2E86C1')
[bar.set_alpha(0.3) for bar in bars]  # Gör errorbars mer genomskinliga
[cap.set_alpha(0.3) for cap in caps]
plt.xlabel('Antal rovdjur', fontsize=16)
plt.ylabel('Medelvärde av andel fångade bytesdjur [%]', fontsize=16)
ax.tick_params(axis='both', labelsize=16, right=True, top=True)


plt.show()
