import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

    def train(self, print_info=True):
        '''
        calculate mean and standard deviation in training data for each features
        mean and std follow the order [Sepal_length, Sepal_width, Petal_length, Petal_width]
        :return: each classes' mean and std as a dictionary
        '''
        dict = {}
        if print_info:
            print('--- mean, standard deviation ---')
            print('order:[Sepal_length, Sepal_width, Petal_length, Petal_width]')
        for i in range(self.num_class):
            data = self.training_data.copy()
            extract = data[data[:, 4] == i]
            mean = np.sum(extract[:, :4], axis=0) / extract[:, :4].shape[0]
            std = (np.sum((extract[:, :4] - mean) ** 2, axis=0) / extract.shape[0]) ** 0.5
            dict[self.CLASSES[i]] = {'mean': mean, 'std': std}
            if print_info:
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


def plot_hist(testing_data, predict):
    '''
    plot results as histogram
    when error occur, title information will be red color
    '''
    for i in range(testing_data.shape[0]):
        plt.figure(i, figsize=(7, 7))
        plt.bar(CLASSES, height=testing_posterior[i, :])
        if testing_data[i, 4] == predict[i]:
            plt.title(
                '[{}/{}] label:'.format(i + 1, testing_data.shape[0]) + CLASSES[
                    int(testing_data[
                            i, 4])] + '\npass!\n' + 'Sepal_length:{:.2f}, Sepal_width:{:.2f}\nPetal_length:{:.2f}, Petal_width:{:.2f}'.format(
                    testing_data[i, 0], testing_data[i, 1], testing_data[i, 2], testing_data[i, 3]),
                loc='left', color='green')
        else:
            plt.title(
                '[{}/{}] label:'.format(i + 1, testing_data.shape[0]) + CLASSES[
                    int(testing_data[
                            i, 4])] + '\nerror!\n' + 'Sepal_length:{:.2f}, Sepal_width:{:.2f}\nPetal_length:{:.2f}, Petal_width:{:.2f}'.format(
                    testing_data[i, 0], testing_data[i, 1], testing_data[i, 2], testing_data[i, 3]),
                loc='left', color='red')
        plt.xlabel('category')
        plt.ylabel('probability')
        plt.show()


def visualize(training_data, testing_data, predict):
    '''
    visualize the results in the distribution of training data
    '''
    for j in range(testing_data.shape[0]):
        plt.figure(j + 1, figsize=(16, 9))

        if testing_data[j, 4] == predict[j]:
            plt.suptitle(
                '[{}/{}] label:'.format(j + 1, testing_data.shape[0]) + CLASSES[int(testing_data[j, 4])] + '\npass',
                color='green')
        else:
            plt.suptitle(
                '[{}/{}] label:'.format(j + 1, testing_data.shape[0]) + CLASSES[
                    int(testing_data[j, 4])] + '\npredict:' + CLASSES[predict[j]] + '\nerror',
                color='red')
        for i in range(4):
            train_info = iris_map.train(print_info=False)
            mu_x = train_info[CLASSES[0]]['mean'][i]
            std_x = train_info[CLASSES[0]]['std'][i]
            mu_y = train_info[CLASSES[1]]['mean'][i]
            std_y = train_info[CLASSES[1]]['std'][i]
            mu_z = train_info[CLASSES[2]]['mean'][i]
            std_z = train_info[CLASSES[2]]['std'][i]

            N = 600
            x = np.linspace(-1, 10, N)
            y = np.linspace(-1, 10, N)
            z = np.linspace(-1, 10, N)
            plt.subplot(4, 4, i * 5 + 1)
            plt.xlim(-1, 10)
            if i == 0:
                plt.ylabel(features[i])
            elif i == 3:
                plt.xlabel(features[i])
            plt.plot(x, stats.norm.pdf(x, mu_x, std_x), color='r')
            plt.plot(y, stats.norm.pdf(y, mu_y, std_y), color='g')
            plt.plot(z, stats.norm.pdf(z, mu_z, std_z), color='b')

            plt.plot(testing_data[j, i], stats.norm.pdf(testing_data[j, i], mu_x, std_x), marker='s', color='r')
            plt.plot(testing_data[j, i], stats.norm.pdf(testing_data[j, i], mu_y, std_y), marker='x', color='g')
            plt.plot(testing_data[j, i], stats.norm.pdf(testing_data[j, i], mu_z, std_z), marker='+', color='b')
        for row in range(4):
            for col in range(4):
                subplot_num = 4 * row + 1 + col
                if subplot_num == 1 or subplot_num == 6 or subplot_num == 11 or subplot_num == 16:
                    continue
                plt.subplot(4, 4, subplot_num)
                if row == 3:
                    plt.xlabel(features[col])
                if col == 0:
                    plt.ylabel(features[row])
                plt.xlim(-1, 10)
                plt.scatter(training_data[training_data[:, 4] == 0][:, col],
                            training_data[training_data[:, 4] == 0][:, row],
                            marker='.', color='r', label=CLASSES[0])
                plt.scatter(training_data[training_data[:, 4] == 1][:, col],
                            training_data[training_data[:, 4] == 1][:, row],
                            marker='.', color='g', label=CLASSES[1])
                plt.scatter(training_data[training_data[:, 4] == 2][:, col],
                            training_data[training_data[:, 4] == 2][:, row],
                            marker='.', color='b', label=CLASSES[2])

                plt.scatter(testing_data[j, col], testing_data[j, row], marker='v', color='k', label='predict')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.figlegend(by_label.values(), by_label.keys(), loc='right')
        plt.show()


if __name__ == '__main__':
    training_data, testing_data = dataloader()
    iris_map = iris_MAP(training_data=training_data, num_class=num_class, CLASSES=CLASSES, features=features)
    # iris_map.test()
    testing_posterior = iris_map.posterior(testing_data[:, :4])
    predict = np.argmax(testing_posterior, axis=1)
    print('--- error rate ---')
    print('training error rate:', np.sum(training_data[:, 4] != predict) / training_data.shape[0])
    print('testing error rate:', np.sum(testing_data[:, 4] != predict) / testing_data.shape[0])

    visualize(training_data, testing_data, predict)
    # plot_hist(testing_data, predict)
