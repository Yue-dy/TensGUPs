import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()


dense_layers = [0]
layer_sizes = [64]
conv_layers = [2]
row = 28
col = 28
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, row,col)
    x_test = x_test.reshape(x_test.shape[0], 1, row,col)
    input_size = (1,row,col)
else:
    x_train = x_train.reshape(x_train.shape[0], row,col,1)
    x_test = x_test.reshape(x_test.shape[0], row,col,1)
    input_size = (row,col,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_test /=255
x_train/=255
fliters = [128,64,64,32,32]
k_sizes = [3,5,4,3,4,5]
num_Conv = 4
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten())
    # matrix ==> one array as input to NN
for i in range(num_Conv-1):
    model.add(Conv2D(fliters[i], (k_sizes[i], k_sizes[i]), padding = "same", activation = tf.nn.relu, input_shape=input_size))
    model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

cur_time = time.time()
model.fit(x_train, y_train,batch_size = 128, epochs=3)
tm = time.time()
print("\n********Time used for training :", tm - cur_time)
cur_time = tm

            
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
        

model.save('MultiGPUs.model')