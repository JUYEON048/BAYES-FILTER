import numpy as np
import matplotlib.pyplot as plt
#from numpy.linalg import inv


#np.random.seed(1)

class Config():
    t = 0.01
    A = np.array([[1, t, (t**2)/2, 0,   0,        0],
                  [0,   1,      t, 0,   0,        0],
                  [0,   0,        1, 0,   0,        0],
                  [0,   0,        0, 1, t, (t**2)/2],
                  [0,   0,        0, 0,   1,      t],
                  [0,   0,        0, 0,   0,        1]])

    b = np.array([[(t**2)/2, t, 1, 0, 0, 0],
                 [0,0,0,(t**2)/2, t, 1]])
    B = b.transpose()
    C = np.array([[1.0,0.0,0.0,0.0,0.0,0.0]])
    Rw = 1  # control input noise
    Rv = 1  # measurement noise



def main():
    print("[START ] Kalman filter Assignment ")
    matrix = Config()
    dt = 0.01
    t = np.arange(0, 3, dt)
    Nsamples = len(t)
    Rsaved = np.zeros([6, Nsamples])
    xestold = np.zeros([6, 1])
    xestnew = np.zeros([6, 1])
    Sensor_data = np.zeros([1, Nsamples])
    Xest = np.zeros([6, Nsamples])
    xhat = np.zeros([6, 1])

    u = np.random.randn(2, Nsamples)
    xold = [0, 0.1, 0, 0, 0.1, 0]

    for n in range(Nsamples):
        xnew = matrix.A.dot(xold) + matrix.B.dot(u[:,n])
        sensor_data = np.sqrt(xnew[0]**2 + xnew[3]**2) + np.random.rand(1,1)*matrix.Rv

        xold = xnew
        Rsaved[:,n] = xnew
        Sensor_data[:,n] = sensor_data[0][0]


    covestold = np.eye(6)
    Xmea = []
    xestold[:, 0] = Rsaved[:, 0]


    for n in range(Nsamples): #Nsamples

        xhat = matrix.A @ xestold
        covhat = (matrix.A @ covestold @ matrix.A.T) + (matrix.B * (matrix.Rw**2) @ np.eye(2) @ matrix.B.T)

        matrix.C[0,0] = xhat[0][0] / np.sqrt(xhat[0][0]**2 + xhat[2][0]**2)
        matrix.C[0,3] = xhat[3][0] / np.sqrt(xhat[0][0]**2 + xhat[3][0]**2)

        K = covhat @ matrix.C.T/(matrix.C @ covhat @ matrix.C.T + np.eye(1) * (matrix.Rv**2))
        xestnew = xhat + K * (Sensor_data[:,n][0] - np.sqrt(xhat[0][0]**2 + xhat[3][0]**2))
        covestnew = (np.eye(6) - (K @ matrix.C)) @ covhat

        xestold = xestnew
        covestold = covestnew
        Xest[:,n] = xestnew[0]
        Xmea.append(np.sqrt(xhat[0][0]**2 + xhat[3][0]**2))

    graph_show(Xest, Rsaved, Xmea, Sensor_data,t)
    print("[FINISH] Kalman filter Assignment ")



def graph_show(Xest, Rsaved, Xmea, Sensor_data, t):

    plt.figure()
    plt.title('Estimated result - "(X,Y)"')
    plt.scatter(Rsaved[0,:], Rsaved[3,:], label='Real')
    plt.scatter(Xest[0,:], Xest[3,:], label='Estimated')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    plt.figure()
    plt.title('Estimated result - "Z"')
    plt.plot(t, Sensor_data[0], 'c', label='Measured')
    plt.plot(t, Xmea, 'b-', label='Estimated')
    plt.xlabel('Time [Sec]')
    plt.ylabel('distance [m]')
    plt.legend(loc='upper left')

    '''plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(Rsaved[:, 1], 'c', label='Real')
    plt.plot(Xest[:,1], 'b', label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel('Time [Sec]')
    plt.ylabel('Predicted_xdot')

    plt.subplot(2, 2, 3)
    plt.plot(Rsaved[:, 2], 'c', label='Real')
    plt.plot(Xest[:,2], 'b', label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel('Time [Sec]')
    plt.ylabel('Predicted_xdotdot')

    plt.subplot(2, 2, 2)
    plt.plot(Rsaved[:, 4], 'c', label='Real')
    plt.plot(Xest[:,4], 'b', label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel('Time [Sec]')
    plt.ylabel('Predicted_ydot')

    plt.subplot(2, 2, 4)
    plt.plot(Rsaved[:, 5], 'c', label='Real')
    plt.plot(Xest[:,5], 'b', label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel('Time [Sec]')
    plt.ylabel('Predicted_ydotdot')'''
    plt.show()



if __name__ == "__main__":
    main()