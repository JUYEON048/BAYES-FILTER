from scipy import io
from sympy import eye
import numpy as np
import matplotlib.pyplot as plt


class Config():
    t = 0.1
    A = np.array([[1, 0.1, (t**2)/2],[0, 1, 0.1],[0, 0, 1]])
    B = np.array([[(t**2)/2], [t], [1]])
    C = np.array([[1, 0, 0]])
    R = 0.1


def main():
    print("[START ] Kalman filter Assignment ")
    avg_before = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cov_before = np.array([[0], [0], [0]])
    result = []

    data_fromMatfile = io.loadmat('./Data.mat')
    for sensor_data in data_fromMatfile['Sensor_data']:
        for data in sensor_data:
            avg, cov = Kalman_filter(avg_before, cov_before, data)
            avg_before, sum_cov = avg, cov
            result.append(avg_before[0][0])

    ref_data = data_fromMatfile['Ref_data']
    ref = [data for data in ref_data[0]]
    plt.plot(ref, 'b', result, 'r')
    plt.show()
    print("[FINISH] Kalman filter Assignment ")


def Kalman_filter(avg_before, cov_before, z):
    matrix = Config()

    avg_predict = np.dot(matrix.A, avg_before)
    cov_predict = matrix.A * cov_before * matrix.A.T + matrix.R

    K = np.dot(np.dot(cov_predict, matrix.C.T), np.linalg.inv(np.dot(np.dot(matrix.C, cov_predict), matrix.C.T) + matrix.R))
    avg = avg_predict + np.dot(K, (z - np.dot(matrix.C, avg_predict)))
    cov = np.dot((eye(3) - np.dot(K, matrix.C)), cov_predict)
    return avg, cov


if __name__ == "__main__":
    main()