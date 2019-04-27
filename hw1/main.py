import scipy.io as scio
import numpy as np

training = scio.loadmat('./training.mat')
testing = scio.loadmat('./testing.mat')
num_class = 3
CLASSES = ('setosa', 'versicolor', 'virginica')
features = ('Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width')


class iris_MAP:
    def __init__(self, training_data, num_class, CLASSES, features):
        self.training_data = training_data
        self.num_class = num_class
        self.CLASSES = CLASSES
        self.features = features
        self.num_data = self.training_data.shape[0]  # total numbers of data

    def likelihood(self, x, mean, std):
        '''
        likelihood estimation with normal distribution
        :param x: numpy.array(k,4) input data
        :param mean: numpy.array(4,) mean for each features
        :param std: numpy.array(4,) standard deviation for each features
        :return: likelihood for each data
        '''
        a = np.exp(((x - mean) ** 2) / (-2 * std ** 2))
        b = 1 / ((2 * np.pi) ** 0.5 * std)
        prob = a * b
        return np.prod(prob, axis=1)

    def prior(self):
        '''
        calculate prior probability for each class
        '''
        prior = {}
        print('--- prior ---')
        for i in range(self.num_class):
            data = self.training_data.copy()
            prior[self.CLASSES[i]] = data[data[:, 4] == i].shape[0] / self.num_data
            print(self.CLASSES[i], '>>>', prior[self.CLASSES[i]])

        return prior

    def train(self):
        '''
        calculate mean and standard deviation in training data for each features
        mean and std follow the order [Sepal_length, Sepal_width, Petal_length, Petal_width]
        :return: each classes' mean and std as a dictionary
        '''
        dict = {}
        print('--- mean, standard deviation ---')
        print('order:[Sepal_length, Sepal_width, Petal_length, Petal_width]')
        for i in range(self.num_class):
            data = self.training_data.copy()
            extract = data[data[:, 4] == i]
            mean = np.sum(extract[:, :4], axis=0) / extract[:, :4].shape[0]
            std = (np.sum((extract[:, :4] - mean) ** 2, axis=0) / extract.shape[0]) ** 0.5
            dict[self.CLASSES[i]] = {'mean': mean, 'std': std}
            print(self.CLASSES[i], '>>>', dict[self.CLASSES[i]])
        return dict

    def posterior(self, x):
        '''
        P(class k | features) = P(class k && features) / P(features)
        N : numbers of data
        :param x: numpy.array(N, 4)
        :return: posterior probability for each classes
        '''
        dict = self.train()
        prior = self.prior()
        likelihood = np.zeros((x.shape[0], self.num_class), dtype='float32')
        prior_prob = np.zeros((1, self.num_class), dtype='float32')
        for i in range(self.num_class):
            likelihood[:, i] = self.likelihood(x, dict[self.CLASSES[i]]['mean'], dict[self.CLASSES[i]]['std'])
            prior_prob[0, i] = prior[self.CLASSES[i]]
        x_prob = np.sum(likelihood * prior_prob ** self.num_class, axis=1)
        posterior_prob = (likelihood * prior_prob ** self.num_class) / x_prob[:, None]
        return posterior_prob

    def test(self):  # my validate program
        x = np.array([[5.1, 3.5, 1.4, 0.2],  # setosa
                      [4.9, 3.0, 1.4, 0.2],  # setosa
                      [7.0, 3.2, 4.7, 1.4],  # versicolor
                      [6.3, 2.8, 5.1, 1.5]])  # virginica
        predict = np.argmax(self.posterior(x), axis=1)
        for i in range(len(x)):
            print(self.CLASSES[predict[i]])


def dataloader():
    '''
    N : numbers of data
    :return: training data numpy.array(N, 5)
             testing data numpy.array(N, 5)
             first 4 columns are features, and the last column is label
    '''
    training_data = np.zeros((training['label'].shape[1], len(features) + 1), dtype='float32')
    for i in range(len(features)):
        training_data[:, i] = training[features[i]]
    training_data[:, -1] = training['label']

    testing_data = np.zeros((testing['label_test'].shape[1], len(features) + 1), dtype='float32')
    for i in range(len(features)):
        testing_data[:, i] = testing[features[i] + '_test']
    testing_data[:, -1] = testing['label_test']
    return training_data, testing_data


training_data, testing_data = dataloader()
iris_map = iris_MAP(training_data=training_data, num_class=num_class, CLASSES=CLASSES, features=features)
# iris_map.test()
predict = np.argmax(iris_map.posterior(testing_data[:, :4]), axis=1)
print('--- error rate ---')
print('training error rate:', np.sum(training_data[:, 4] != predict) / training_data.shape[0])
print('testing error rate:', np.sum(testing_data[:, 4] != predict) / testing_data.shape[0])
