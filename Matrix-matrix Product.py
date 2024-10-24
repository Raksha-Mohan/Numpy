import numpy as np
import time
n = 100
A = np.random.rand(n, n)
B = np.random.rand(n, n)
C = np.zeros((n, n))
CC = np.zeros((n, n))

# For loop
start_loop = time.perf_counter()
for i in range(n):
    for j in range(n):
        sum_value = 0
        for k in range(n):
            sum_value += A[i, k] * B[k, j]
        C[i, j] = sum_value
end_loop = time.perf_counter()
tl = end_loop - start_loop
print('loop time:', tl)

# Partial vectorization
start_pvt = time.perf_counter()
for j in range(n):
    CC[:, j] = np.dot(A, B[:, j])
end_pvt = time.perf_counter()
pvt = end_pvt - start_pvt
print('Partial vectorization time:', pvt)

# Full vectorization
start_vt = time.perf_counter()
CCC = np.dot(A, B)
end_vt = time.perf_counter()
vt = end_vt- start_vt
print('Vectorization time:', vt)

# Comparing the results
print("Difference in the results (C and CC):", np.linalg.norm(C - CC))
print("Difference in the results (CC and CCC):", np.linalg.norm(CC - CCC))

#Speedup
print('Speedup', tl/pvt)
print('Speedup1', tl/vt)
print('Speedup2', pvt/vt)