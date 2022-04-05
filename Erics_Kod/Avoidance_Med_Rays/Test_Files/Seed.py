import numpy as np
seed = 0
N = 2
rng = np.random.default_rng(seed)
print(rng.random(N))
print(rng.uniform(-1/2,1/2,N))
print(rng.uniform(-1/2,1/2,N))


rng = np.random.default_rng(seed)
print(rng.random(N))
print(rng.uniform(-1/2,1/2,N))
print(rng.uniform(-1/2,1/2,N))

seed = [i for i in range(N)]
print(seed)

