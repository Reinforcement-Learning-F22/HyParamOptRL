from keras import models
from keras import layers
import tensorflow as tf
import os

tf.keras.backend.set_floatx('float32')
tf.keras.backend.set_image_data_format('channels_first')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelDefectRecognition:

    def __init__(self, **kwargs):

        input_function = kwargs.pop('activation_input')
        output_function = kwargs.pop('activation_output')
        optimizer = kwargs.pop('optimizer')

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(16, (7, 7), activation=input_function, input_shape=(4, 150, 128)))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Conv2D(12, (5, 5), activation=input_function))
        self.model.add(layers.MaxPool2D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(24, activation=input_function))
        self.model.add(layers.Dense(4, activation=output_function))

        self.model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.Precision()])

    def fit(self, x_train, y_train):
        print('x_train shape: ', x_train.shape)
        self.model.fit(x_train, y_train, epochs=15)

    def score(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test)
        print(score)
        return self.model.evaluate(x_test, y_test)[1]
