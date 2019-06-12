import numpy as np
from sklearn.decomposition import PCA

from Dataloader import Data
from model import NN
from utils.plot import plot_info
from utils.plot import plot_decision_region

train_path = './Data_train'
test_path = './Data_test'
EPOCHS = 300
batch_size = 512
lr = 0.01
pca_components = 2


def to_onehot(x, n_cls):
    onehot = np.zeros((x.shape[0], n_cls), dtype=int)
    onehot[range(x.shape[0]), x] = 1
    return onehot


def main():
    train_loader = Data(train_path, shuffle=True, split=0.2)
    pca = train_loader.dataloading(batch_size=batch_size, pca=PCA(n_components=pca_components))
    test_loader = Data(test_path, shuffle=False, test=True)
    test_loader.dataloading(pca=pca)

    model = NN(in_dim=pca_components + 1, n_cls=3, neurons=[256], lr=lr, hidden_activation='sigmoid',
               load_weight=False, save_weight=False)
    info = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'EPOCHS': EPOCHS}
    for epoch in range(EPOCHS):
        cache = np.zeros(4)
        itr = 0
        for train_data in train_loader:
            x_train, y_train, x_val, y_val = train_data
            y_train = to_onehot(y_train, 3)
            y_val = to_onehot(y_val, 3)
            train_loss, train_acc = model.train(x_train, y_train)
            val_loss, val_acc, pred = model.predict(x_val, y_val)
            cache[0] += train_loss
            cache[1] += train_acc
            cache[2] += val_loss
            cache[3] += val_acc
            itr += 1
        cache /= itr
        info['train_loss'].append(cache[0])
        info['train_acc'].append(cache[1])
        info['val_loss'].append(cache[2])
        info['val_acc'].append(cache[3])
        print('EPOCH:{:05d}/{:05d}  train_loss: {:.5f}  train_acc: {:.4f}  val_loss: {:.5f}  val_acc: {:.4f}'.format(epoch + 1, EPOCHS, *cache.tolist()))
    plot_info(**info)
    train_loader.set_batch_size(2000)
    for train_data in train_loader:
        x_train, y_train, x_val, y_val = train_data
        y_train = to_onehot(y_train, 3)
        plot_decision_region(x_train, y_train, model, train_loader.CLASSES, title='training')

    for test_data in test_loader:
        x_test, y_test = test_data
        y_test = to_onehot(y_test, 3)
        plot_decision_region(x_test, y_test, model, train_loader.CLASSES, title='testing')
        test_loss, test_acc, pred = model.predict(x_test, y_test)
    print('test_loss: {:.5f} test_acc: {:.4f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    main()
