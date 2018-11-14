import numpy as np


def norm(array):
    return np.expand_dims(np.sqrt(np.sum(np.square(array), -1)), -1)


embedding = np.load('../model_data/glove_embedding.npy')
vehicles = [0, 1, 3, 5, 11, 16, 19]
animals = [2, 6, 8, 10, 14, 17]

x = embedding[animals]
print(np.dot(x, x.T) / (np.dot(norm(x), norm(x).T)))
