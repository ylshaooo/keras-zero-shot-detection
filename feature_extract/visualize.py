import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def reduce_dim():
    # load features
    res_car = np.load('data/resnet_feat_car.npy')
    res_dog = np.load('data/resnet_feat_dog.npy')
    res_sofa = np.load('data/resnet_feat_sofa.npy')
    res_train = np.load('data/resnet_feat_train.npy')
    yolo_car = np.load('data/yolo_feat_car.npy')
    yolo_dog = np.load('data/yolo_feat_dog.npy')
    yolo_sofa = np.load('data/yolo_feat_sofa.npy')
    yolo_train = np.load('data/yolo_feat_train.npy')

    x_resnet = np.concatenate([res_car, res_dog, res_sofa, res_train], 0)
    x_yolo = np.concatenate([yolo_car, yolo_dog, yolo_sofa, yolo_train], 0)

    # reduce dimensions for visualization
    print('Fitting data...')
    xr = TSNE(learning_rate=100).fit_transform(x_resnet)
    xy = TSNE(learning_rate=100).fit_transform(x_yolo)

    print('Finish training.')
    np.save('data/tsne_resnet.npy', xr)
    np.save('data/tsne_yolo.npy', xy)


def plot():
    # numbers: car-822, dog-1031, sofa-139, train-482
    xr = np.load('data/tsne_resnet.npy')
    xy = np.load('data/tsne_yolo.npy')

    # plot clustering results
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    p1 = plt.scatter(xr[:822, 0], xr[:822, 1], marker='.', alpha=0.5)
    p2 = plt.scatter(xr[822:1853, 0], xr[822:1853, 1], marker='.', alpha=0.5)
    p3 = plt.scatter(xr[1853:1992, 0], xr[1853:1992, 1], marker='.', alpha=0.5)
    p4 = plt.scatter(xr[1992:, 0], xr[1992:, 1], marker='.', alpha=0.5)
    plt.legend([p1, p2, p3, p4], ['car', 'dog', 'sofa', 'train'])
    plt.axis([-80, 80, -80, 80])
    plt.subplot(122)
    p1 = plt.scatter(xy[:822, 0], xy[:822, 1], marker='.', alpha=0.5)
    p2 = plt.scatter(xy[822:1853, 0], xy[822:1853, 1], marker='.', alpha=0.5)
    p3 = plt.scatter(xy[1853:1992, 0], xy[1853:1992, 1], marker='.', alpha=0.5)
    p4 = plt.scatter(xy[1992:, 0], xy[1992:, 1], marker='.', alpha=0.5)
    plt.legend([p1, p2, p3, p4], ['car', 'dog', 'sofa', 'train'])
    plt.axis([-80, 80, -80, 80])
    fig.savefig('results.png')


if __name__ == '__main__':
    # reduce_dim()
    plot()
