import matplotlib.pyplot as plt
import numpy as np
import time

# Q1 = np.load('fish_eaten_matrix_Q1.npy')
# Q2 = np.load('fish_eaten_matrix_Q2.npy')
# Q3 = np.load('fish_eaten_matrix_Q3.npy')
# Q4 = np.load('fish_eaten_matrix_Q4.npy')

Q1 = np.array([[1,1],[2,2]])
Q2 = np.array([[3,3],[4,4]])
Q3 = np.array([[5,5],[6,6]])
Q4 = np.array([[7,7],[8,8]])

Q1Q2 = np.concatenate( (Q1, Q2),axis=1)
Q3Q4 = np.concatenate( (Q3, Q4),axis=1)
Q = np.concatenate( (Q1Q2,Q3Q4), axis= 0)
print(Q1)
print()
print(Q2)
print()
print(Q3)
print()
print(Q4)
print()
print(Q1Q2)
print()
print(Q3Q4)
print()
print(Q)





