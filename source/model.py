import numpy as np
import tensorflow as tf

from models.inception_v1 import Inception_V1 

class Model:
    # 모델 생성자
    def __init__(self, model_type, learning_rate, class_num):
        
        # model parameters
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # model types 
        if model_type == 'Inception_V1': 
            self.net = Inception_V1(num_class=self.class_num, use_auxlayer=True)
            self.net.build(input_shape=(None, 224, 224, 3))
            self.net.summary()

    # 학습 함수
    @tf.function
    def train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            y_pred, aux1_pred, aux2_pred = self.net(x, training=True)
            y_pred = tf.squeeze(y_pred)
            metric = self.compute_metric(x, y_true, y_pred, aux1_pred, aux2_pred)
        
        grads = tape.gradient(metric['loss'], self.net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

        return metric

    # 지표 계산 함수
    def compute_metric(self, x, y_true, y_pred, aux1_pred, aux2_pred):
        
        metric = dict()
        metric['pred'] = tf.argmax(y_pred, axis=1)
        metric['loss'] = self.compute_loss_mult(y_true, y_pred, aux1_pred, aux2_pred) 

        return metric

    def compute_loss_mult(self, y_true, y_pred, aux1_pred, aux2_pred):
        y_onehot = tf.one_hot(y_true, self.class_num)
        output_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, y_pred))

        aux1_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, aux1_pred))
        aux2_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_onehot, aux2_pred))

        loss = output_loss + 0.3*(aux1_loss+aux2_loss)

        return loss


    def save(self, path):
        self.net.save_weights(path)

    def load(self, path):
        self.net.load_weights(path)

    
