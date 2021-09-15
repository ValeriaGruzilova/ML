import numpy as np
from matplotlib import pyplot as plt


def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке

    return 1 / (2 * m) * np.sum((compute_hypothesis(X, theta) - y) ** 2)


def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    # n = X.shape[1]  # количество признаков с фиктивным
    for i in range(num_iter):
        theta_temp = theta.copy()
        theta_temp = theta_temp - alpha * (compute_hypothesis(X, theta) - y).dot(X) / m
        theta = theta_temp
        # =====================

        history.append(compute_cost(X, y, theta))
    return history, theta


def scale_features(X):
    m = X.shape[0]  # количество примеров в выборке

    mj = np.sum(X, axis=0) / m  # мат ожидание
    sigma = np.sqrt(1 / (m - 1) * np.sum((X - mj) ** 2, axis=0))  # ско

    X_0 = X[:, 0]
    X_0 = X_0[:, np.newaxis]  # приведение к вектору-столбцу

    return np.concatenate([X_0, np.divide((X[:, 1:] - mj[1:]), sigma[1:])], axis=1)


def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data('lab1data2.txt')

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации до нормализации')
plt.plot(range(len(history)), history)
plt.show()

X = scale_features(X)

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации после нормализации')
plt.plot(range(len(history)), history)
plt.show()

theta_solution = normal_equation(X, y)
print(f'theta, посчитанные через градиентный спуск: {np.round(theta, 3)}, '
      f'\ntheta через через нормальное уравнение:     {np.round(theta_solution, 3)}')
