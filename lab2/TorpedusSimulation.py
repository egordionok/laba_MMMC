from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import OceanForm
import oceanwidget
import math
import numpy as np
import random as rd
from matplotlib.animation import FuncAnimation
import sympy as sp
import pprint
import time
import scipy.io as io
import pickle
#from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure

class OceanWidget(QMainWindow, OceanForm.Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        self.setupUi(self)

        self.setWindowTitle("Водная гладь")

        self.pushButton.clicked.connect(self.HereAreWeGo)
        #self.StopButton.clicked.connect(self.HereAreWeGo)
        #self.F_Bar.valueChanged.connect(self.F_valuechange)
        #self.Angle_Bar.valueChanged.connect(self.Angle_valuechange)
        #self.addToolBar(NavigationToolbar(self.PlotWidget.canvas, self))





    def HereAreWeGo(self):
        self.widget.canvas.axes.clear()
        global X, Y, Phi, Vx, Vy, Omega, cl, cb, cv, R, rho, a, b, Vo, J, m, X_aim, Y_aim, Phi_aim, Vx_aim, Vy_aim, Omega_aim, Explode, T_Explode, dt, Step_o, Detected, cl_aim, cb_aim, cv_aim, S_front, rho, S_side, L, Vo_aim, R_scr, S_r, J_aim, m_aim, X_mind, Y_mind, Vx_mind, Vy_mind, Phi_mind, monevr
        dt = 0.01
        k=1
        Explode = 0
        T_Explode = 0
        Detected = 0
        Alpha_max = 0.3
        R_s = 30
        G_s = 0.7

        cl = float(self.Cl_Edit.text())
        cb = float(self.Cb_Edit.text())
        cv = float(self.Cv_Edit.text())
        R = float(self.R_Edit.text())
        rho = float(self.Rho_Edit.text())
        a = float(self.a_Edit.text())
        b = float(self.b_Edit.text())
        Vo = float(self.V0_Edit.text())
        m = float(self.M_Edit.text())
        J = (1 / 6) * m * (a ** 2 + b ** 2)
        R = float(self.R_Edit.text())
        print(2)
        X = float(self.X0_Edit.text())
        Y = float(self.Y0_Edit.text())
        Phi = float(self.Phi0_Edit.text())
        Vx = float(self.Vx0_Edit.text())
        Vy = float(self.Vy0_Edit.text())
        Omega = float(self.Omega0_Edit.text())

        print(2.4)
        cl_aim = float(self.Cl_Edit_2.text())
        cb_aim = float(self.Cb_Edit_2.text())
        cv_aim = float(self.Cv_Edit_2.text())
        S_front = float(self.S_f_Edit.text())
        S_side = float(self.S_s_Edit.text())
        L = float(self.L_Edit.text())
        R_scr = float(self.R_scr_Edit.text())
        S_r = float(self.S_r_Edit.text())
        Vo_aim = float(self.V0_Edit_2.text())
        m_aim = float(self.M_Edit_2.text())
        J_aim = (1 / 2) * m_aim * (L ** 2)
        print(2)
        X_aim = float(self.X0_Edit_2.text())
        Y_aim = float(self.Y0_Edit_2.text())
        Phi_aim = float(self.Phi0_Edit_2.text())
        Vx_aim = float(self.Vx0_Edit_2.text())
        Vy_aim = float(self.Vy0_Edit_2.text())
        Omega_aim = float(self.Omega0_Edit_2.text())
        print(2.6)
        self.widget.canvas.axes.axis('equal')
        self.widget.canvas.axes.set(xlim=[-100, 100], ylim=[-100, 100])
        print(2.8)
        TorpedaX = np.array([-a * 0.66, b * 0.8, b, b * 0.8, -a * 0.66, -a * 0.8, -a, -a, -0.8 * a, -0.66 * a])
        TorpedaY = np.array([R, R, 0, -R, -R, R, R, -R, -R, R])
        DrawnTorpeda = self.widget.canvas.axes.plot(TorpedaX, TorpedaY)[0]
        print(3)
        #self.widget.canvas.axes.plot(x,y)
        print(4)
        AimX = np.array([-L/2,-L/5,L/6,L/3,L/2,L/3,L/6,-L/5,-L/2,-L/2])
        AimY = np.array([0.9*L/6, L/6, 0.9*L/6, 0.5*L/6, 0, -0.5*L/6, -0.9*L/6, -L/6, -0.9*L/6, 0.9*L/6])
        print(4.5)
        Aim = self.widget.canvas.axes.plot(AimX,AimY)[0]
        #self.widget.canvas.axes.axis('scaled')
        self.widget.canvas.show()
        N_o = 10
        Phi_o = np.linspace(0,6.28, N_o)
        np.hstack([np.cos(Phi_o), np.cos(Phi_o) * 0.8])
        X_o = np.hstack([np.hstack([np.cos(Phi_o), np.cos(Phi_o+0.5*6.28/(N_o-1))*0.8]), np.hstack([np.cos(Phi_o)*0.6, np.cos(Phi_o+0.5*6.28/(N_o-1))*0.4])])
        Y_o = np.hstack([np.hstack([np.sin(Phi_o), np.sin(Phi_o+0.5*6.28/(N_o-1))*0.8]), np.hstack([np.sin(Phi_o)*0.6, np.sin(Phi_o+0.5*6.28/(N_o-1))*0.4])])
        Step_o = 0.005
        print(5)

        X_mind, Y_mind, Vx_mind, Vy_mind, Phi_mind = X_aim, Y_aim, Vx_aim, Vy_aim, Phi_aim
        # Vx_mind, Vy_mind = 0, 0
        monevr = 0
        znak_povorota = 0
        print(6)

        def Rot2D(X, Y, Phi):  # rotates point (X,Y) on angle alpha with respect to Origin
            RX = X * np.cos(Phi) - Y * np.sin(Phi)
            RY = X * np.sin(Phi) + Y * np.cos(Phi)
            return RX, RY

        def MoveEquations(x, y, phi, Vx, Vy, Omega, alpha):
            global cl, cb, cv, R, rho, a, b, Vo, J, m
            Vl = Vx * np.cos(phi) + Vy * np.sin(phi)
            Vb = Vy * np.cos(phi) - Vx * np.sin(phi)

            Fl = cl * 3.14 * R ** 2 * abs(Vl) * Vl * rho / (2)
            Fb = 2 * cb * R * (a + b) * abs(Vb) * Vb * rho / (2)
            Ms = 2 * cv * R * (a + b) * abs(Omega) * Omega * rho / (2)
            Fdv = 3.14 * R ** 2 * Vo ** 2 * rho
            Fr = 3.14 * (R * 0.9) ** 2 * Vo ** 2 * rho * np.sin(alpha) / 2
            dX = Vx
            dY = Vy
            dPhi = Omega
            dVx = (-Fl * np.cos(phi) + Fb * np.sin(phi) + Fdv * np.cos(phi) + Fr * np.sin(phi - alpha)) / m
            dVy = (-Fl * np.sin(phi) - Fb * np.cos(phi) + Fdv * np.sin(phi) - Fr * np.cos(phi - alpha)) / m
            dOmega = (Fr * a * np.cos(alpha) - Ms) / J
            return dX, dY, dPhi, dVx, dVy, dOmega

        def AimMoveEquations(x_aim, y_aim, phi_aim, Vx_aim, Vy_aim, Omega_aim, alpha_aim):
            # print('V', Vx_aim, Vy_aim)
            """
            Fl - лобовое
            Fb - боковое
            Fdv - двигателя
            Fr - руля
            Ms - момент сопротивления
            :param x_aim:
            :param y_aim:
            :param phi_aim:
            :param Vx_aim:
            :param Vy_aim:
            :param Omega_aim:
            :param alpha_aim:
            :return:
            """
            global cl_aim, cb_aim, cv_aim, S_front, rho, S_side, L, Vo_aim, R_scr, S_r, J_aim, m_aim, rho
            Vl = Vx_aim * np.cos(phi_aim) + Vy_aim * np.sin(phi_aim)
            Vb = Vy_aim * np.cos(phi_aim) - Vx_aim * np.sin(phi_aim)

            Fl = cl_aim * S_front * abs(Vl) * Vl * rho / (2)
            Fb = cb_aim * S_side * abs(Vb) * Vb * rho / (2)
            Ms = cv_aim * S_side * abs(Omega_aim) * Omega_aim * rho / (2)
            Fdv = 3.14 * R_scr ** 2 * Vo_aim ** 2 * rho
            Fr = S_r * Vl ** 2 * rho * np.sin(alpha_aim) / 2
            dX = Vx_aim
            dY = Vy_aim
            dPhi = Omega_aim
            dVx = (-Fl * np.cos(phi_aim) + Fb * np.sin(phi_aim) + Fdv * np.cos(phi_aim) + Fr * np.sin(phi_aim - alpha_aim)) / m_aim
            dVy = (-Fl * np.sin(phi_aim) - Fb * np.cos(phi_aim) + Fdv * np.sin(phi_aim) - Fr * np.cos(phi_aim - alpha_aim)) / m_aim
            dOmega = (Fr * L/2 * np.cos(alpha_aim) - Ms) / J_aim
            return dX, dY, dPhi, dVx, dVy, dOmega

        def NewPoints(i):
            global X, Y, Phi, Vx, Vy, Omega, cl, cb, cv, R, rho, a, b, Vo, J, m, X_aim, Y_aim, Phi_aim, Vx_aim, Vy_aim, Omega_aim, Explode, T_Explode, dt, Step_o, Detected, cl_aim, cb_aim, cv_aim, S_front, rho, S_side, L, Vo_aim, R_scr, S_r, J_aim, m_aim, X_mind, Y_mind, Vx_mind, Vy_mind, Phi_mind, monevr, znak_povorota
            # t+=dt

            if Explode == 0:
                Alpha_aim = float(self.AngleSlider.value())/400
                TCx = X_aim - X
                TCy = Y_aim - Y

                # Координаты в памяти торпеды
                DeltaX = X_mind - X
                DeltaY = Y_mind - Y

                # print(Vx_mind, Vy_mind)
                gamma = np.arctan2(DeltaY, DeltaX) - np.arctan2(np.sin(Phi), np.cos(Phi))
                while abs(gamma) > math.pi:
                    gamma -= 2 * math.pi * np.sign(gamma)

                if DeltaX**2 + DeltaY**2 < a**2:
                    X_mind, Y_mind, Vx_mind, Vy_mind, Phi_mind = X_aim, Y_aim, Vx_aim, Vy_aim, Phi_aim
                    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                    monevr = 1
                    DrawnTorpeda.set_color([1, 0, 0])
                    if gamma > 0:
                        znak_povorota = -1
                    else:
                        znak_povorota = 1


                if TCx**2 + TCy**2 < a**2:
                    Explode = 1

                N_TCx = TCx/(TCx**2 + TCy**2)
                N_TCy = TCy/(TCx**2 + TCy**2)
                N_Tx = np.cos(Phi)
                N_Ty = np.sin(Phi)
                beta = np.arctan2(TCy, TCx) - np.arctan2(np.sin(Phi), np.cos(Phi))
                while abs(beta) > math.pi:
                    beta -= 2*math.pi*np.sign(beta)

                if monevr == 1:
                    X_mind, Y_mind, Vx_mind, Vy_mind, Phi_mind = X_aim, Y_aim, Vx_aim, Vy_aim, Phi_aim
                    # Координаты в памяти торпеды
                    DeltaX = X_mind - X
                    DeltaY = Y_mind - Y

                    gamma = np.arctan2(DeltaY, DeltaX) - np.arctan2(np.sin(Phi), np.cos(Phi))
                    while abs(gamma) > 2 * math.pi:
                        gamma -= 2 * math.pi * np.sign(gamma)

                    if abs(gamma) > math.pi / 3:
                        if znak_povorota == 1:
                            Alpha = Alpha_max
                        else :
                            Alpha = -Alpha_max
                    else:
                        if k * gamma > Alpha_max:
                            Alpha = Alpha_max
                        elif k * gamma < -Alpha_max:
                            Alpha = -Alpha_max
                        else:
                            Alpha = k * gamma

                    # gamma += math.pi
                    #
                    # print(gamma)
                    # if gamma > math.pi / 4 and gamma < 7 / 4 * math.pi:
                    #     gamma -= math.pi
                    #     gamma = -gamma
                    #     if k * gamma > Alpha_max:
                    #         Alpha = Alpha_max
                    #     elif k * gamma < Alpha_max:
                    #         Alpha = -Alpha_max
                    #     else:
                    #         Alpha = k * gamma
                    # else:
                    #     gamma -= math.pi
                    #     if k * gamma > Alpha_max:
                    #         Alpha = Alpha_max
                    #     elif k * gamma < -Alpha_max:
                    #         Alpha = -Alpha_max
                    #     else:
                    #         Alpha = k * gamma

                    # if k * gamma > Alpha_max:
                    #     Alpha = Alpha_max
                    # elif k * gamma < -Alpha_max:
                    #     Alpha = -Alpha_max
                    # else:
                    #     Alpha = k * gamma
                else:
                    if k * gamma > Alpha_max:
                        Alpha = Alpha_max
                    elif k * gamma < -Alpha_max:
                        Alpha = -Alpha_max
                    else:
                        Alpha = k * gamma



                if abs(beta) < G_s and TCx**2 + TCy**2<R_s**2:
                    Detected = 1
                    # DrawnTorpeda.set_color([1, 0, 0])
                    # if k*beta>Alpha_max:
                    #     Alpha = Alpha_max
                    # elif k*beta<-Alpha_max:
                    #     Alpha = -Alpha_max
                    # else:
                    #     Alpha = k*beta
                else:

                    Detected = 0
                    # DrawnTorpeda.set_color([0, 0, 1])
                dX, dY, dPhi, dVx, dVy, dOmega = MoveEquations(X, Y, Phi, Vx, Vy, Omega, Alpha)
                X += dX * dt
                Y += dY * dt
                Phi += dPhi * dt
                Vx += dVx * dt
                Vy += dVy * dt
                Omega += dOmega * dt
                RTX, RTY = Rot2D(TorpedaX, TorpedaY, Phi)
                DrawnTorpeda.set_data(X + RTX, Y + RTY)
                dX_aim, dY_aim, dPhi_aim, dVx_aim, dVy_aim, dOmega_aim = AimMoveEquations(X_aim, Y_aim, Phi_aim, Vx_aim, Vy_aim, Omega_aim, Alpha_aim)
                X_aim += dX_aim * dt
                Y_aim += dY_aim * dt
                Phi_aim += dPhi_aim * dt
                Vx_aim += dVx_aim * dt
                Vy_aim += dVy_aim * dt

                dX_mind, dY_mind, dPhi_mind, dVx_mind, dVy_mind, _ = AimMoveEquations(X_mind, Y_mind, Phi_mind, Vx_mind, Vy_mind, 0, 0)
                X_mind += dX_mind * dt
                Y_mind += dY_mind * dt
                Vx_mind += dVx_mind * dt
                Vy_mind += dVy_mind * dt
                # print('V_mind', Vx_mind, Vy_mind)
                Phi_mind += dPhi_mind * dt

                Omega_aim += dOmega_aim * dt
                RAX, RAY = Rot2D(AimX, AimY, Phi_aim)
                Aim.set_data(X_aim + RAX, Y_aim + RAY)
            else:
                T_Explode += Step_o
                Step_o = 0.9*Step_o
                DrawnTorpeda.set_color([1,0,0])
                DrawnTorpeda.set_marker('o')
                DrawnTorpeda.set_data(X_o*a*T_Explode/dt+X, Y_o*a*T_Explode/dt+Y)


            return [DrawnTorpeda,Aim]
        fig = self.widget.canvas.figure
        animation = FuncAnimation(fig, NewPoints, interval=dt*1000, blit=True, frames=1001)
        self.widget.canvas.draw()





app = QApplication([])
window = OceanWidget()
window.show()
app.exec_()