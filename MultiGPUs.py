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
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test,y_test) = mnist.load_data()

row = 28
col = 28
num_categ = 10
num_col_chennals = 1 
# i for gray_scale

if tf.keras.backend.image_data_format() == 'channels_first':
    # the channel dimension for the image data default by last
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

y_train = tf.keras.utils.to_categorical(y_train, num_categ)
y_test = tf.keras.utils.to_categorical(y_test, num_categ)

fliters = [128,64,64,32,32,16]
k_sizes = [3,5,4,3,4,5]
num_Conv = 4

mirrored_stategy = tf.distribute.MirroredStrategy() 
# pick all gpus by defult
with mirrored_stategy.scope():
    model = tf.keras.models.Sequential()
    for i in range(num_Conv):
        print(i)
        model.add(Conv2D(fliters[i], (k_sizes[i], k_sizes[i]), padding = "same", activation = tf.nn.relu, input_shape=input_size))
        model.add(MaxPooling2D(pool_size=(2,2)))
        # each iteration the edge size -2 shouldn't let it < 1 

    model.add(Flatten())
    # martix ==> array as input

    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_categ,activation = 'softmax'))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

    cur_time = time.time()
    model.fit(x_train, y_train,batch_size = 128,  epochs=3)
    tm = time.time()
    print("\n********Time used for training :", tm - cur_time)
    cur_time = tm

            
    score = model.evaluate(x_test,y_test,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

        

model.save('MultiGPUs.model')