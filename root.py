import numpy as np
import time
from joblib import Parallel, delayed
from scipy.optimize import fsolve

# Function to find the root using numpy for array handling
def findroot(x) :
    f =np.sin(3 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * x))
    return f

a = -3
b = 5
n = 4 ** 9
result=[]

x0 = np.linspace(a, b, n)

# Serial Computation (Normal Loop)
s_t = time.perf_counter()
res_s = [fsolve(findroot, x) for x in x0]
ts = time.perf_counter() - s_t
print("ts:", ts)

# Parallel Computation
workers=2
p_t = time.perf_counter()
res_p = Parallel(n_jobs=workers)(delayed(fsolve)(findroot, x) for x in x0)
tp = time.perf_counter() - p_t
print("tp:", tp)

# Calculating Speedup and Efficiency
speedup = ts / tp
efficiency = speedup / workers*100  # Adjusted to the number of parallel jobs
print("Speedup:", speedup)
print("Efficiency:", efficiency)
