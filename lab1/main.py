import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = 100

VG = 2
NG = -2
X = (np.random.sample(n) * 2 * (VG - 0.1) + NG + 0.1)
Y = (np.random.sample(n) * 2 * (VG - 0.1) + NG + 0.1)
R = 3

dt = 0.01
r = 0.05
k = 1.1

Vx = R * (1 - 2 * np.random.sample(n))
Vy = R * (1 - 2 * np.random.sample(n))

fig = plt.figure(figsize=[13, 9])
ax = fig.add_subplot(111)
ax.axis('equal')
DrawnPoints = [ax.plot(x, y, marker='o')[0] for x, y in zip(X, Y)]

ax.clear()
ax.plot([-2, -2, 2, 2, -2], [-2, 2, 2, -2, -2], color='black')


def MoveEquations(X, Y, Vx, Vy):
    dX = Vx
    dY = Vy
    dVx = 0
    dVy = -9.81
    return dX, dY, dVx, dVy


def NewPoints(i):
    global X, Y, Vx, Vy, DrawnPoints
    dX, dY, dVx, dVy = MoveEquations(X, Y, Vx, Vy)
    X += dX * dt
    Y += dY * dt
    Vx += dVx * dt
    Vy += dVy * dt

    for i in range(n):
        for j in range(n):
            if j == i:
                continue

            Nx = X[i] - X[j]
            Ny = Y[i] - Y[j]
            N = (Nx) ** 2 + (Ny) ** 2
            if N < 4 * r ** 2:
                Vx1p = (-Ny * Vy[i] * Nx + Ny ** 2 * Vx[i] + Nx ** 2 * Vx[j] + Ny * Nx * Vy[j]) / N
                Vy1p = (Nx ** 2 * Vy[i] - Nx * Ny * Vx[i] + Ny * Nx * Vx[j] + Ny ** 2 * Vy[j]) / N
                Vx2p = (-Ny * Nx * Vy[j] + Ny ** 2 * Vx[j] + Nx ** 2 * Vx[i] + Ny * Vy[i] * Nx) / N
                Vy2p = (Nx ** 2 * Vy[j] - Ny * Nx * Vx[j] + Nx * Ny * Vx[i] + Ny ** 2 * Vy[i]) / N

                Vx[i] = Vx1p
                Vy[i] = Vy1p
                Vx[j] = Vx2p
                Vy[j] = Vy2p

    for i in range(n):
        if X[i] > (VG - r) or X[i] < (NG + r):
            X[i] -= dX[i] * dt
            Vx[i] = -Vx[i]*k
            X[i] += Vx[i] * dt

        if Y[i] > (VG - r) or Y[i] < (NG + r):
            Y[i] -= dY[i] * dt
            Vy[i] = -Vy[i]*k
            Y[i] += Vy[i] * dt

        if X[i] > (VG - r):
            X[i] = (VG - r)
        if X[i] < (NG + r):
            X[i] = (NG + r)
        if Y[i] > (VG - r):
            Y[i] = (VG - r)
        if Y[i] < (NG + r):
            Y[i] = (NG + r)

    for x, y, P in zip(X, Y, DrawnPoints):
        P.set_data(x, y)
    return DrawnPoints


a = FuncAnimation(fig, NewPoints, interval=dt * 1000, blit=True)
plt.show()
