import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


dt = 0.01
global cl, cb, cv, R, rho, a, b, Vo, J, m
Ts = np.linspace(0,1000,30000)
Alphas = 0.3*np.sin(Ts)
cl = 0.47
cb = 0.52
cv = 0.7
R = 0.3
rho = 1000
a = 2
b = 1
Vo = 10
m = 2000
J = (1/6)*m*(a**2+b**2)

X = 0
Y = 0
Phi = 0
Vx = 0
Vy = 0
Omega = 0
Alpha = 0.1
fig = plt.figure(figsize=[13,9])
ax = fig.add_subplot(111)
ax.axis('equal')
ax.set(xlim=[-100,100], ylim=[-100,100])
TorpedaX = np.array([-a*0.66, b*0.8, b, b*0.8, -a*0.66, -a*0.8, -a, -a, -0.8*a, -0.66*a])
TorpedaY = np.array([R, R, 0, -R, -R, R, R, -R, -R, R])
DrawnTorpeda = ax.plot(TorpedaX,TorpedaY)[0]

def Rot2D(X, Y, Phi):#rotates point (X,Y) on angle alpha with respect to Origin
    RX = X*np.cos(Phi) - Y*np.sin(Phi)
    RY = X*np.sin(Phi) + Y*np.cos(Phi)
    return RX, RY

def MoveEquations(x,y,phi,Vx,Vy,Omega,alpha):
    global cl, cb, cv, R, rho, a, b, Vo, J, m
    Vl = Vx * np.cos(phi)+Vy * np.sin(phi)
    Vb = Vy * np.cos(phi)-Vx * np.sin(phi)

    Fl = cl*3.14*R**2*abs(Vl)*Vl*rho/(2)
    Fb = 2 * cb * R * (a + b) *abs(Vb)* Vb * rho / (2)
    Ms = 2 * cv * R * (a + b) * abs(Omega)*Omega * rho / (2)
    Fdv = 3.14 * R**2 * Vo**2 * rho
    Fr = 3.14 * (R*0.9)**2 * Vo**2 * rho * np.sin(alpha)/2
    dX = Vx
    dY = Vy
    dPhi = Omega
    dVx = (-Fl*np.cos(phi)+Fb*np.sin(phi)+Fdv*np.cos(phi)+Fr*np.sin(phi-alpha))/m
    dVy = (-Fl*np.sin(phi)-Fb*np.cos(phi)+Fdv*np.sin(phi)-Fr*np.cos(phi-alpha))/m
    dOmega = (Fr*a*np.cos(alpha)-Ms)/J
    return dX,dY,dPhi,dVx,dVy,dOmega

def NewPoints(i):
   global X, Y, Phi, Vx, Vy, Omega, cl, cb, cv, R, rho, a, b, Vo, J, m
   #t+=dt
   Alpha = Alphas[i]
   dX,dY,dPhi,dVx,dVy,dOmega = MoveEquations(X,Y,Phi,Vx,Vy,Omega,Alpha)
   X += dX*dt
   Y += dY*dt
   Phi += dPhi * dt
   Vx += dVx * dt
   Vy += dVy * dt
   Omega += dOmega * dt
   RTX, RTY = Rot2D(TorpedaX,TorpedaY,Phi)
   DrawnTorpeda.set_data(X+RTX,Y+RTY)

   return [DrawnTorpeda]

anima = FuncAnimation(fig, NewPoints, interval=dt*1000, blit=True)
plt.show()