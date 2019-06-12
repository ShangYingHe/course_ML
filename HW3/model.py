import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(dout, x):
    return dout * (sigmoid(x) * (1 - sigmoid(x)))


def relu(x):
    return np.fmax(x, 0)


def relu_backward(dout, x):
    return np.sign(x) * dout


def softmax(x):
    return (np.exp(x.T) / np.sum(np.exp(x.T), axis=0)).T


def crossentropy(y, yhat):
    return np.sum(- np.log(np.sum(y * yhat, axis=1))) / y.shape[0]


def fc_forward(x, w, b):
    y = x.dot(w) + b
    return y


def fc_backward(dout, x, w, b):
    dx = dout.dot(w.T)
    dw = (x.T).dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def accuracy(y, yhat):
    return np.sum(np.argmax(y, axis=1) == np.argmax(yhat, axis=1)) / yhat.shape[0]


activation = {'sigmoid': [sigmoid, sigmoid_backward], 'relu': [relu, relu_backward]}


class NN:
    def __init__(self, in_dim, n_cls, neurons=[256], lr=0.01, hidden_activation='sigmoid', load_weight=True, save_weight=False):
        '''
        :param in_dim: input dimension
        :param n_cls: number of class
        :param neurons: a list of neurons for hidden layer
        :param lr: learning rate
        '''
        self.in_dim = in_dim
        self.n_hidden_layers = len(neurons)
        self.n_cls = n_cls
        self.lr = lr
        self.activate = activation[hidden_activation][0]
        self.activate_backward = activation[hidden_activation][1]
        self.load_weight = load_weight
        self.save_weight = save_weight
        self.neurons = neurons
        self.w = {}
        self.b = {}
        self.mu = 0.0
        self.sigma = 0.5
        for i in range(self.n_hidden_layers + 1):
            if i == 0:
                self.w[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.in_dim, neurons[i]))
                self.b[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(neurons[i]))
            elif i == self.n_hidden_layers:
                self.w[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(neurons[-1], self.n_cls))
                self.b[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.n_cls))
            else:
                self.w[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(neurons[i - 1], neurons[i]))
                self.b[str(i)] = np.random.normal(loc=self.mu, scale=self.sigma, size=(neurons[i]))
            if self.load_weight:
                try:
                    self.w[str(i)] = np.load('./weight/w_' + str(i) + '.npy')
                    self.b[str(i)] = np.load('./weight/b_' + str(i) + '.npy')
                except:
                    continue
    def up_grad(self, dw, db):
        self.lr = self.lr * 0.999
        for i in range(len(self.w)):
            self.w[str(i)] -= dw[i] * self.lr
            self.b[str(i)] -= db[i] * self.lr
            if self.save_weight:
                np.save('./weight/w_'+str(i)+'.npy', self.w[str(i)])
                np.save('./weight/b_'+str(i)+'.npy', self.b[str(i)])

    def train(self, x, yhat):
        # --------- forward --------- #
        y = []
        for i in range(self.n_hidden_layers + 1):
            if i == 0:
                y.append(fc_forward(x, self.w[str(i)], self.b[str(i)]))
                y[0] = self.activate(y[0])
            elif i == self.n_hidden_layers:
                y.append(fc_forward(y[-1], self.w[str(i)], self.b[str(i)]))
                y[-1] = softmax(y[-1])
            else:
                y.append(fc_forward(y[i - 1], self.w[str(i)], self.b[str(i)]))
                y[i] = self.activate(y[i])
        loss = crossentropy(y[-1], yhat)
        acc = accuracy(y[-1], yhat)
        # --------- backward --------- #
        dw = []
        db = []
        douty = (y[-1] - yhat) / x.shape[0]
        for i in range(self.n_hidden_layers, 0, -1):
            douty, _dw, _db = fc_backward(douty, y[i - 1], self.w[str(i)], self.b[str(i)])
            douty = self.activate_backward(douty, y[i - 1])
            dw.append(_dw)
            db.append(_db)
        douty, _dw, _db = fc_backward(douty, x, self.w['0'], self.b['0'])
        dw.append(_dw)
        db.append(_db)
        dw.reverse()
        db.reverse()
        self.up_grad(dw, db)
        y.clear()
        dw.clear()
        db.clear()
        return loss, acc

    def predict(self, x, yhat=None):
        y = []
        for i in range(self.n_hidden_layers + 1):
            if i == 0:
                y.append(fc_forward(x, self.w[str(i)], self.b[str(i)]))
                y[0] = self.activate(y[0])
            elif i == self.n_hidden_layers:
                y.append(fc_forward(y[-1], self.w[str(i)], self.b[str(i)]))
                y[-1] = softmax(y[-1])
            else:
                y.append(fc_forward(y[i - 1], self.w[str(i)], self.b[str(i)]))
                y[i] = self.activate(y[i])
        if yhat is not None:
            loss = crossentropy(y[-1], yhat)
            acc = accuracy(y[-1], yhat)
            return loss, acc, y[-1]
        else:
            return y[-1]
