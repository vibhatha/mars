import numpy as np
import mars.tensor as mt
import time

CN = 1000000000.0


def simple():
    N = 200_000_000

    a = np.random.uniform(-1, 1, size=(N, 2))
    t1 = time.time_ns()
    npy_norm = np.linalg.norm(a, axis=1)
    t2 = time.time_ns()
    npy_less = npy_norm < 1
    t3 = time.time_ns()
    npy_ag = npy_less.sum() * 4 / N
    t4 = time.time_ns()

    print(f"Numpy Time : Norm = {(t2 - t1) / CN}, "
          f"Less Than 1 = {(t3 - t2) / CN}, "
          f"Aggregation = {(t4 - t3) / CN},"
          f"Total = {(t4 - t1) / CN}")

    a = mt.random.uniform(-1, 1, size=(N, 2))

    t1 = time.time_ns()
    mt_norm = mt.linalg.norm(a, axis=1)
    t2 = time.time_ns()
    mt_less = mt_norm < 1
    t3 = time.time_ns()
    mt_ag = mt_less.sum() * 4 / N
    mt_ag.execute()
    t4 = time.time_ns()

    # print(((mt.linalg.norm(a, axis=1) < 1)
    #        .sum() * 4 / N).execute())
    print(f"Mars Time : Norm = {(t2 - t1) / CN}, "
          f"Less Than 1 = {(t3 - t2) / CN}, "
          f"Aggregation = {(t4 - t3) / CN}, "
          f", Total = {(t4 - t1) / CN}")


def matmul():
    row = 30_000
    col = 2

    a = np.arange(row * col)
    b = np.reshape(a, [row, col])
    c = np.reshape(a, [col, row])
    t1 = time.time_ns()
    d = np.matmul(b, c)
    sum = d.sum()
    print(f"Numpy Mat Mul Time [{row}] x [{row}] => SUM {sum}, Time = {(time.time_ns() - t1) / CN}")

    a = mt.arange(row * col)
    b = mt.reshape(a, [row, col])
    c = mt.reshape(a, [col, row])
    t1 = time.time_ns()
    d: mt = mt.matmul(b, c)
    sum = d.sum().execute()
    print(f"Mars Mat Mul Time [{row}] x [{row}] => SUM {sum}, Time = {(time.time_ns() - t1) / CN}")


def scalar_mul():
    row = 100_000_000
    col = 2

    a = np.arange(row * col)
    b = np.reshape(a, [row, col])
    t1 = time.time_ns()
    d = b * 2
    sum = d.sum()
    print(f"Numpy Scalar Mul Time [{row}] x [{col}] => SUM {sum}, Time = {(time.time_ns() - t1) / CN}")

    a = mt.arange(row * col)
    b = mt.reshape(a, [row, col])
    t1 = time.time_ns()
    d = b * 2
    sum = d.sum().execute()
    print(f"Mars Scalar Mul Time [{row}] x [{col}] => SUM {sum}, Time = {(time.time_ns() - t1) / CN}")


def transpose():
    row = 100_000_000
    col = 10

    a = np.arange(row * col)
    b = np.reshape(a, [row, col])
    t1 = time.time_ns()
    d = b.T
    print(f"Numpy Mat Transpose Time [{row}] x [{col}] => SUM {d.shape}, Time = {(time.time_ns() - t1) / CN}")

    a = mt.arange(row * col)
    b = mt.reshape(a, [row, col])
    t1 = time.time_ns()
    d = b.T
    e = d.execute()
    print(f"Mars Mat Transpose Time [{row}] x [{col}] => SUM {e.shape}, Time = {(time.time_ns() - t1) / CN}")


transpose()