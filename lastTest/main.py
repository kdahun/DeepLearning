import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, Dropout, MaxPool2D
from keras.layers import Dense,Flatten, Activation
from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os,glob


obs_Xdata=[[[0,0,1,0,0],
           [0,1,0,1,0],
           [0,1,0,1,0],
           [1,0,0,0,1],
           [1,0,0,0,1]],
           [[1,1,1,1,0],
            [1,0,0,0,1],
            [1,1,1,1,0],
            [1,0,0,0,1],
            [1,1,1,1,0]],
           [[0,1,1,1,1],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [0,1,1,1,1]],
           [[1,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,1,1,1,0]],
           [[0,1,1,1,1],
            [1,0,0,0,0],
            [1,1,1,1,1],
            [1,0,0,0,0],
            [0,1,1,1,1]]]
obs_Ylable=[0,1,2,3,4]

imagedata = np.array(obs_Xdata)
imagelabel = np.array(obs_Ylable)
#
# train_images = obs_Xdata[:]
# train_labels= obs_Ylable[:]
#
imagedata = imagedata.astype('float32') / 255.0
# # 원핫인코딩
imagelabel = to_categorical(imagelabel, num_classes=5)
# # 원핫 인코딩과 softmax 개수 맞춰야함.

# 층 구성후 모델 학습
model = Sequential()


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
######
model.add(Dense(5, activation='softmax'))
# ######
# model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(imagedata, imagelabel, epochs=10, batch_size=1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'])
plt.show()








