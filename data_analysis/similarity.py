import numpy as np


def cosine_similarity(tensor0, tensor1, axis=-1):
    tensor0_norm = np.sqrt(np.sum(np.square(tensor0), axis=axis))
    tensor1_norm = np.sqrt(np.sum(np.square(tensor1), axis=axis))
    inner_prod = np.sum(tensor0 * tensor1, axis=axis) / (tensor0_norm * tensor1_norm)
    return inner_prod


total = 0
x = np.zeros([20])
embedding = np.load('model_data/glove_embedding.npy')

with open('model_data/train.txt') as f:
    lines = f.readlines()

for line in lines:
    info = line.strip().split(' ')
    if len(info) == 1:
        print(info)
        print(line)
    boxes = info[1]
    boxes = boxes.split(',')
    for i in range(4, len(boxes), 5):
        cls = int(boxes[i])
        x += cosine_similarity(embedding, np.expand_dims(embedding[cls], 0))
        total += 1

print(x / total)
