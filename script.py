import numpy as np

A = np.array([[1, 2, 3], [3, 4, 5]])
B = np.array([[5, 6], [7, 8], [9, 10]])

print("A\n-->\n", A)
print("B\n-->\n", B)

print(A @ B)
