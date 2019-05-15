import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

training_data = pd.read_csv('./Training_set.csv', header=None)
testing_data = pd.read_csv('./Testing_set.csv', header=None)


class linear_regression:
    def __init__(self, x_train, y_train, O1, O2, alpha):
        self.x_train = x_train
        self.y_train = y_train
        self.O1 = O1
        self.O2 = O2
        self.alpha = alpha

        self.feature_vector_len = self.O1 * self.O2 + 2
        self.m_0 = np.zeros(self.feature_vector_len)
        self.s_0 = 1 / self.alpha * np.identity(self.feature_vector_len)
        self.beta = 1 / 0.28
        self.prior = multivariate_normal(mean=self.m_0, cov=self.s_0)

        self.m_N = self.m_0
        self.s_N = self.s_0
        self.posterior = self.prior

    def phi(self, x):
        phi_x = np.ones((x.shape[0], self.O2 * self.O1 + 2))
        s_1 = (np.max(self.x_train[:, 0]) - np.min(self.x_train[:, 0])) / (self.O1 - 1)
        s_2 = (np.max(self.x_train[:, 1]) - np.min(self.x_train[:, 1])) / (self.O2 - 1)

        for i in range(1, self.O1 + 1):
            for j in range(1, self.O2 + 1):
                k = self.O2 * (i - 1) + j
                u_i = s_1 * (i - 1)
                u_j = s_2 * (j - 1)
                a = ((x[:, 0] - u_i) ** 2) / (2 * s_1 ** 2)
                b = ((x[:, 1] - u_j) ** 2) / (2 * s_2 ** 2)
                phi_x[:, k - 1] = np.exp(-a - b)
        phi_x[:, -2] = x[:, 2]
        phi_x[:, -1] = 1
        return phi_x

    def update_posterior(self, x, t, i):
        if x.ndim == 1:
            x = x[None, :]
        phi = self.phi(x)
        if i == 0:
            self.s_N = np.linalg.inv(np.linalg.inv(self.s_0) + self.beta * phi.T.dot(phi))
            self.m_N = self.s_N.dot(np.linalg.inv(self.s_0).dot(self.m_0[:, None]) + self.beta * phi.T.dot(t))
        else:
            a = self.s_N
            b = self.m_N
            self.s_N = np.linalg.inv(np.linalg.inv(self.s_0) + self.beta * phi.T.dot(phi))
            self.m_N = self.s_N.dot(np.linalg.inv(self.s_0).dot(self.m_0[:, None]) + self.beta * phi.T.dot(t))
            self.s_0 = a
            self.m_0 = b.flatten()
        self.posterior = multivariate_normal(mean=self.m_N.flatten(), cov=self.s_N)

    def bayesian_train(self):
        for i in range(self.x_train.shape[0]):
            self.update_posterior(self.x_train[i, :], self.y_train[i], i)
        return self.m_N, self.s_N

    def bayesian_predict(self, x):
        x = x[None, :]
        phi = self.phi(x).T
        x_mean = self.m_N[:, None].T.dot(phi)
        x_sigma_sq = 1 / self.beta + phi.T.dot(self.s_N).dot(phi)
        return x_mean



def data_split(training_data, testing_data):
    x_train = training_data[:, :3]
    y_train = training_data[:, 3]
    x_test = testing_data[:, :3]
    y_test = testing_data[:, 3]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_split(training_data.values, testing_data.values)
    O1 = 2
    O2 = 2
    alpha = 25.0
    linear_reg = linear_regression(x_train, y_train, O1, O2, alpha)
    m_N, s_N = linear_reg.bayesian_train()
    bayesian_pred = np.zeros(y_test.shape[0])
    for i in range(x_test.shape[0]):
        bayesian_pred[i] = linear_reg.bayesian_predict(x_test[i, :])
    sq_error = (y_test - bayesian_pred) ** 2
    MSE = np.sum(sq_error) / y_test.shape[0]
    pd.DataFrame({'predict': bayesian_pred, 'label': y_test, 'square error': sq_error}).to_csv('./result.csv', index=0)
    print('mean square error:', MSE)
