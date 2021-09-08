import os
import time
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf 

from sklearn.utils import shuffle

from model import Model
from utils.read_data import read_data


class_num = 10
class_dict = {
    0 : 'airplane',
    1 : 'bird',
    2 : 'car',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'horse',
    7 : 'monkey',
    8 : 'ship',
    9 : 'truck'
}

epochs = 1000
batch_size = 32
lr = 0.001
model = Model(model_type='Inception_V1', learning_rate=lr, class_num=class_num)

im_dir = './img/'
img_names = []
labels = []
for key in class_dict.keys():
    files = os.listdir(os.path.join(im_dir, str(key+1)))

    for file in files:
        img_names.append(os.path.join(im_dir, str(key+1), file))
        labels.append(key)

img_names, labels = shuffle(img_names, labels)


loss = tf.keras.metrics.Mean() 
losses = []

for epoch in range(epochs):
    print(f'\n{epoch+1}/{epochs} Epochs Start!')
    start_time_epoch = time.time()

    # for mode in ['train', 'val']: 
    #     if epoch % 5 != 0 and mode == 'val':
    #         continue

    dataset = tf.data.Dataset.from_tensor_slices((img_names, labels))
    dataset = dataset.map(read_data, num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size)

    # total iterator of batches in dataset
    epoch_iters = int(np.ceil(len(img_names) / batch_size))
    # loop through all batches 
    for batch_num, batch in enumerate(dataset):
        start_time_batch = time.time()
        xs, ys = batch[0], batch[1]
        
        metrics = model.train_step(xs, ys)

        zs = metrics['pred']
        loss(metrics['loss'])
        
        print(f'{batch_num+1}/{epoch_iters} pred : {zs} label : {ys}  loss: {loss.result()}\r', end = '')

    losses.append(loss.result())


