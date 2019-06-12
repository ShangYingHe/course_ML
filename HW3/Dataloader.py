import os
import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, path, num_cls=None, split=0.0, shuffle=True, test=False):
        self.path = path
        if num_cls is not None:
            self.num_cls = num_cls
        else:
            self.num_cls = len(os.listdir(self.path))
        self.CLASSES = []
        self.split = split
        self.shuffle = shuffle
        self.test = test
        self.data_idx = 0
        self.batch_size = 8
        self.x = None
        self.y = None
        self.batch_x = None
        self.batch_y = None
        self.x_val = None
        self.y_val = None

    def set_batch_size(self, batch_size):
        if batch_size > self.x.shape[0]:
            self.batch_size = self.x.shape[0]
            print('batch size =', self.x.shape[0])
        else:
            self.batch_size = batch_size
            print('batch size =', batch_size)
        self.data_idx = 0

    def _shuffle(self):
        shuffle_idx = np.random.choice(self.x.shape[0], self.x.shape[0], replace=False)
        self.x = self.x[shuffle_idx, :]
        self.y = self.y[shuffle_idx]

    def split_data(self):
        split_idx = int(self.x.shape[0] * self.split)
        self.x_val = self.x[:split_idx, :]
        self.y_val = self.y[:split_idx]
        self.x = self.x[split_idx:, :]
        self.y = self.y[split_idx:]

    def dataloading(self, batch_size=8, pca=None):
        self.batch_size = batch_size
        x = []
        y = []
        for i, dir in enumerate(os.listdir(self.path)):
            i_dir = os.path.join(self.path, dir)
            self.CLASSES.append(dir)
            for filename in os.listdir(i_dir):
                img = plt.imread(os.path.join(i_dir, filename))
                x.append(np.transpose(img, (2, 0, 1))[0, :, :])
                y.append(i)
        x = np.asarray(x)
        y = np.asarray(y)
        if np.max(x) > 1:
            x /= 255
        if pca is not None and not self.test:
            x = x.reshape((x.shape[0], -1))
            self.x = pca.fit_transform(x, y)
            self.x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1)
            self.y = y
        elif pca is not None and self.test:
            x = x.reshape((x.shape[0], -1))
            self.x = pca.transform(x)
            self.x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1)
            self.y = y
        else:
            self.x = x
            self.y = y

        if self.shuffle:
            self._shuffle()
        if self.split != 0:
            self.split_data()
        if self.batch_size > self.x.shape[0]:
            self.batch_size = self.x.shape[0]
        if self.test:
            self.batch_size = self.x.shape[0]
        if pca is not None and not self.test:
            return pca

    def __len__(self):
        return self.x.shape[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_idx + self.batch_size <= self.x.shape[0]:
            self.batch_x = self.x[self.data_idx:self.data_idx + self.batch_size, :]
            self.batch_y = self.y[self.data_idx:self.data_idx + self.batch_size]
            self.data_idx += self.batch_size
        else:
            dif = self.batch_size - (self.x.shape[0] - self.data_idx)
            self.batch_x = self.x[self.data_idx:, :]
            self.batch_y = self.y[self.data_idx:]
            self.data_idx = 0
            if self.shuffle:
                self._shuffle()
            self.batch_x = np.concatenate((self.batch_x, self.x[:dif, :]), axis=0)
            self.batch_y = np.concatenate((self.batch_y, self.y[:dif]), axis=0)
            self.data_idx += dif
            raise StopIteration
        if self.split != 0:
            return self.batch_x, self.batch_y, self.x_val, self.y_val
        else:
            return self.batch_x, self.batch_y
