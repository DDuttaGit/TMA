from U_support import REPLwithU

# Attention Module
# sm(XAX^T)XV

# Matrix Transposition  (3 x dot)     --> X^T
# Matrix Multiplication (3 x dot x 3) --> Y1 = X . A
# Matrix Multiplication (3 x dot x 3) --> Y2 = Y1 . X^T
# Matrix Softmax (3 x dot x 3)        --> Y3 = sm(Y2)
# Matrix Multiplication (3 x dot x 3) --> Y4 = X . V
# Matrix Multiplication (3 x dot x 3) --> Y5 = Y3 . Y4

A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
V = [0.1, 0.7, 0.4, 1, 0.3, 0.9, 0.5, 0.2, 0.6]
X = [0.8, 0.20, 0.5, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7]

r, c = 3, 3


excecutor = REPLwithU()
excecutor.env.set_variable('X', X)
excecutor.run_given_line('set example X')
excecutor.run_given_line("examples off")
excecutor.run_given_line('XT = Transpose_3dot()(X);')
X_top = excecutor.env.get_variable('XT').val._vals
excecutor.env.set_variable('XA', X+A)
excecutor.run_given_line('Y1 = matmul_3dot3()(XA);')
Y1 = excecutor.env.get_variable('Y1').val._vals[:r*c]
excecutor.env.set_variable('Y1XT', Y1+X_top)
excecutor.run_given_line('Y2 = matmul_3dot3()(Y1XT);')
Y2 = excecutor.env.get_variable('Y2').val._vals[:r*c]
excecutor.env.set_variable('Y2', Y2)
excecutor.run_given_line('Y3 = softmax_3dot()(Y2);')
Y3 = excecutor.env.get_variable('Y3').val._vals
excecutor.env.set_variable('XV', X+V)
excecutor.run_given_line('Y4 = matmul_3dot3()(XV);')
Y4 = excecutor.env.get_variable('Y4').val._vals[:r*c]
excecutor.env.set_variable('Y3Y4', Y3+Y4)
excecutor.run_given_line('Y5 = matmul_3dot3()(Y3Y4);')
Y5 = excecutor.env.get_variable('Y5').val._vals[:r*c]
print(Y5)

from typing import Iterable
import numpy as np


def softmax_rows(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    row_max = np.max(m, axis=1, keepdims=True)
    e = np.exp(m - row_max)
    return e / np.sum(e, axis=1, keepdims=True)


def attention_flat(A_flat: Iterable, X_flat: Iterable, V_flat: Iterable, n: int = 3) -> np.ndarray:
    A = np.asarray(A_flat, dtype=float).reshape((n, n))
    X = np.asarray(X_flat, dtype=float).reshape((n, n))
    V = np.asarray(V_flat, dtype=float).reshape((n, n))

    S = X @ A @ X.T
    S_sm = softmax_rows(S)
    XV = X @ V
    Y = S_sm @ XV
    return Y

Y = attention_flat(A, X, V, n=3)
print("Result (3x3):")
np.set_printoptions(precision=8, suppress=True)
print(Y)
print("From Original Numpy Computation: ", Y.reshape(-1).tolist())
print("From RASP Library Utilization: ", Y5)
y_np = Y.reshape(-1)
y_rasp = np.asarray(list(Y5), dtype=float)

if y_np.shape != y_rasp.shape:
    raise ValueError(f"Shape mismatch: numpy {y_np.shape} vs rasp {y_rasp.shape}")

abs_diff = np.abs(y_np - y_rasp)
max_diff = float(np.max(abs_diff))
mean_diff = float(np.mean(abs_diff))
is_close = np.allclose(y_np, y_rasp, rtol=1e-3, atol=1e-3)

print("Max abs difference:", max_diff)
print("Mean abs difference:", mean_diff)
print("Allclose:", is_close)

if not is_close:
    raise AssertionError(f"Results differ (max abs diff = {max_diff})")