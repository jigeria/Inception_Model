import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model

class Inception_V1(tf.keras.Model):
    def __init__(self, num_class=10, use_auxlayer=False):
        super(Inception_V1, self).__init__()
        self.use_auxlayer = use_auxlayer

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding='same')
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')

        self.conv2 = tf.keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')

        self.inception_3a = Inception_Block(
            filters_1x1=64,
            filters_3x3_reduce=96,
            filters_3x3=128,
            filters_5x5_reduce=16,
            filters_5x5=32,
            filters_pool=32
            )

        self.inception_3b = Inception_Block(
            filters_1x1=128,
            filters_3x3_reduce=128,
            filters_3x3=192,
            filters_5x5_reduce=32,
            filters_5x5=96,
            filters_pool=64
            )

        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same')

        self.inception_4a = Inception_Block(
            filters_1x1=192,
            filters_3x3_reduce=96,
            filters_3x3=208,
            filters_5x5_reduce=16,
            filters_5x5=48,
            filters_pool=64
            )
        
        self.inception_4b = Inception_Block(
            filters_1x1=160,
            filters_3x3_reduce=112,
            filters_3x3=224,
            filters_5x5_reduce=24,
            filters_5x5=64,
            filters_pool=64
            )

        self.inception_4c = Inception_Block(
            filters_1x1=128,
            filters_3x3_reduce=128,
            filters_3x3=256,
            filters_5x5_reduce=24,
            filters_5x5=64,
            filters_pool=64
            )

        self.inception_4d = Inception_Block(
            filters_1x1=112,
            filters_3x3_reduce=144,
            filters_3x3=288,
            filters_5x5_reduce=32,
            filters_5x5=64,
            filters_pool=64
            )

        self.inception_4e = Inception_Block(
            filters_1x1=256,
            filters_3x3_reduce=160,
            filters_3x3=320,
            filters_5x5_reduce=32,
            filters_5x5=128,
            filters_pool=128
            )

        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2), padding='same')

        self.inception_5a = Inception_Block(
            filters_1x1=256,
            filters_3x3_reduce=160,
            filters_3x3=320,
            filters_5x5_reduce=32,
            filters_5x5=128,
            filters_pool=128
            )

        self.inception_5b = Inception_Block(
            filters_1x1=384,
            filters_3x3_reduce=192,
            filters_3x3=384,
            filters_5x5_reduce=48,
            filters_5x5=128,
            filters_pool=128
            )

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)
        self.drop_out = tf.keras.layers.Dropout(rate=0.4)
        self.fc = tf.keras.layers.Dense(units=num_class, activation='linear')

        if use_auxlayer:
            self.aux1 = Inception_Auxiliary_Classifier(num_class=num_class)
            self.aux2 = Inception_Auxiliary_Classifier(num_class=num_class)
        else:
            self.aux1, self.aux2 = None, None


    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool3(x)
        x = self.inception_4a(x)


        if self.use_auxlayer:
            aux1 = self.aux1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        if self.use_auxlayer:
            aux2 = self.aux2(x)

        x = self.inception_4e(x)
        x = self.max_pool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
    
        x = self.avg_pool(x)
        x = self.drop_out(x)
        x = self.fc(x)

        if self.use_auxlayer:
            return x, aux1, aux2
        else:
            return x


class Inception_Auxiliary_Classifier(tf.keras.Model):
    def __init__(self, num_class):
        super(Inception_Auxiliary_Classifier, self).__init__()

        self.conv = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3, padding='same'),
            tf.keras.layers.Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same')]
        )

        self.fc = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1024, activation='linear'),
            tf.keras.layers.Dropout(rate=0.7),
            tf.keras.layers.Dense(units=10, activation='softmax')]
        )


    def call(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


class Inception_Block(tf.keras.Model):
    def __init__(
        self,
        filters_1x1,
        filters_3x3_reduce,
        filters_3x3,
        filters_5x5_reduce,
        filters_5x5,
        filters_pool
        ):
        
        super(Inception_Block, self).__init__()

        self.branch_1 = tf.keras.layers.Conv2D(filters_1x1, kernel_size=(1, 1), activation='relu', padding='same')

        self.branch_2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters_3x3_reduce, kernel_size=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters_3x3, kernel_size=(3, 3), activation='relu', padding='same')]
        )

        self.branch_3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters_5x5_reduce, kernel_size=(1, 1), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters_5x5, kernel_size=(3, 3), activation='relu', padding='same')]
        )

        self.branch_4 = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=1, padding='same'),
            tf.keras.layers.Conv2D(filters_pool, kernel_size=(1, 1), activation='relu', padding='same')]
        )

    def call(self, x):
        x = tf.concat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], axis=3)

        return x

if __name__=="__main__":
    net = Inception_V1(num_class=10, use_auxlayer=True)
    net.build(input_shape=(None, 224, 224, 3))
    net.summary()
    x = tf.random.normal((3,224,224,3))
    net(x)