import tensorflow as tf
import numpy as np


def read_data(filename, label):
    ''' 
    Takes input of image/label names and creates a nested tensor for data and label. 
    Input
        filename : image patch path
        label : label mask path
    Output
        image : tensor image
        mask : tensor mask
    ''' 
    
    # print('filename', type(filename))

    image_string = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image_string)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    
    return image, label, filename