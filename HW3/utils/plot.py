import numpy as np
import matplotlib.pyplot as plt


def plot_info(train_loss, train_acc, val_loss, val_acc, EPOCHS):
    plt.figure(0, figsize=(15, 5))
    plt.subplot(1, 2, 1)
    x = np.linspace(1, EPOCHS, EPOCHS)
    plt.plot(x, train_loss, color='#F18904', label='train_loss')
    plt.plot(x, val_loss, color='#36688D', label='val_loss')
    plt.xlabel('EPOCH')
    plt.ylabel('loss')
    plt.legend(loc='right')
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc, color='#F18904', label='train_acc')
    plt.plot(x, val_acc, color='#36688D', label='val_acc')
    plt.xlabel('EPOCH')
    plt.ylabel('acc')
    plt.legend(loc='right')
    plt.savefig('./train_info.png')
    plt.close()


def plot_decision_region(x, y, model, CLASSES, title):
    print('plot {} decision region...'.format(title))
    gt_color = ['#ae0000', 'g', '#000079']
    boundary_color = ['#ffd2d2', '#a6ffa6', '#84c1ff']
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    y = np.argmax(y, axis=1)
    step = 600
    plt.figure(0, figsize=(15, 10))
    plt.title(title)
    xx = np.linspace(np.min(x[:, 1]) - 1.5, np.max(x[:, 1]) + 1.5, step)
    yy = np.linspace(np.min(x[:, 0]) - 0, np.max(x[:, 0]) + 0, step)
    xx = xx.repeat(step)
    yy = np.tile(yy, step)
    space = np.concatenate((xx[:, None], yy[:, None], np.ones((step * step, 1))), axis=1)
    y_space = model.predict(space)
    y_space = np.argmax(y_space, axis=1)
    for i in range(3):
        plt.scatter(xx[y_space == i], yy[y_space == i], marker='.', color=boundary_color[i], alpha=0.5, s=18.5)
        plt.scatter(None, None, marker='s', color=boundary_color[i], label='decision region for class ' + CLASSES[i])
    for i_gt in range(3):
        plt.scatter(x[y == i_gt, 0], x[y == i_gt, 1], marker='.', color=gt_color[i_gt], label=CLASSES[i_gt], s=18.5)
        for j_pred in range(3):
            if j_pred == i_gt:
                continue
            idx = np.argwhere((y == i_gt) * (y_pred == j_pred) == 1)
            plt.scatter(x[idx, 0], x[idx, 1], marker='x', color=gt_color[i_gt])
    plt.scatter(None, None, marker='x', color='black', label='predict error')
    plt.legend(loc='upper right')
    nodes = ''
    for node in model.neurons:
        nodes = nodes + str(node) + '_'
    plt.savefig('./decision_region_{}{}.png'.format(title, nodes[:-1]), dpi=100)
    plt.close()
