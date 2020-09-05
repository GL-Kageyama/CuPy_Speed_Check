import numpy as np 
import cupy as cp 
import time

#=============================================================
# Verify the performance difference between NumPy and CuPy
#=============================================================

# Allow a range that exceeds GPU memory
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Size of the matrix
n = 25000

print("====================================================================")

npTimeCheck1 = time.time()

# Calculating inner product of a 25000x25000 matrix using NumPy
np_A = np.random.rand(n, n)
np_B = np.random.rand(n, n)
np_Results = np.dot(np_A, np_B)

npTimeCheck2 = time.time()

print(np_Results)

print("--------------------------------------------------------------------")

npTimeCheck3 = time.time()

print("NumPy")
print("Dot product calculation time : ", npTimeCheck2 - npTimeCheck1, "second")
print("Total time to output         : ", npTimeCheck3 - npTimeCheck1, "second")

print("====================================================================")

cpTimeCheck1 = time.time()

# Calculating inner product of a 5000x25000 matrix using CuPy
cp_A = cp.random.rand(n, n)
cp_B = cp.random.rand(n, n)
cp_Results = cp.dot(cp_A, cp_B)

cpTimeCheck2 = time.time()

print(cp_Results)

print("--------------------------------------------------------------------")

cpTimeCheck3 = time.time()

print("CuPy")
print("Dot product calculation time : ", cpTimeCheck2 - cpTimeCheck1, "second")
print("Total time to output         : ", cpTimeCheck3 - cpTimeCheck1, "second")

print("====================================================================")