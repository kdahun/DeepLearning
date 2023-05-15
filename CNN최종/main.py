

import cv2
import os,glob
import numpy
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np


image_size =(256,256)
image_folder = 2
path=os.path.dirname(os.path.abspath(__file__))
print(path)
path=os.path.dirname('C:\\Users\\dahun\\Desktop\\race\\part1\\')
print(path)

obs_Xdata=[]
obs_Ylable=[]
path_obs=[]
obs_list=[]

for i in range(0,image_folder):
    path_obs=path+'/'+str(i)
    obs_img_file=glob.glob(path_obs+"/*.jpg")
    obs_list.extend(obs_img_file)
    for j in obs_img_file:
        obs_img=cv2.imread(j,cv2.IMREAD_GRAYSCALE)
        obs_img=cv2.resize(obs_img,image_size)
        obs_Xdata.append(obs_img)
        obs_Ylable.append(i)

# for j in range(0,2):
#     plt.figure()
#     plt.imshow(obs_Xdata[j])
#     plt.show()
#     cv2.imshow('Gray Scale Image',obs_Xdata[j])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



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

class myCallback(Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('loss')<0.05:
            print('\n Stop training.')
            self.model.stop_training=True


callbacks=myCallback()

imagedata=np.array(obs_Xdata)
imagelabel=np.array(obs_Ylable)
print(len(imagedata),len(imagelabel))
train_images, test_images,train_labels,test_labels=train_test_split(imagedata,imagelabel,test_size=0.2,shuffle=True)

train_images = train_images.astype('float32')/255.0
test_images=test_images.astype('float32')/255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(256,256,1)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

with tf.device("/device:GPU:0"):
    history = model.fit(train_images,train_labels,epochs=150,batch_size=30,callbacks=[callbacks])
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'])
plt.show()

loss_and_accuracy = model.evaluate(test_images,test_labels)

print('accuray = '+str(loss_and_accuracy[1]))
print('loss = '+ str(loss_and_accuracy[0]))

model.save('obs_model.h5')


new_obs_model=keras.models.load_model('obs_model.h5')
print(obs_list)

obs_y=['0','1']
average_list=[]
count=0
sum=0
path=os.path.dirname('C:\\Users\\dahun\\Desktop\\white\\')
obs_img_file1=glob.glob(path+"/*.jpg")
print(obs_img_file1)
for img,label in zip(test_images,test_labels):
    # frame=cv2.imread(img)
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # frame=cv2.resize(frame,(128,128))
    predicted_result= new_obs_model.predict(np.array([img]))
    obs_predicted = predicted_result[0]
    for i,obsClass in enumerate(obs_predicted):
        print(obs_Ylable[i],'=',int(obsClass*100))
    print('Predicted Result = ',obs_y[obs_predicted.argmax()],"  정답 : ", np.where(label==1)[0][0])
    average_list.append(obs_y[obs_predicted.argmax()])
    plt.imshow(img)
    tmp="Prediction"+obs_y[obs_predicted.argmax()]+"  정답 : ", np.where(label==1)[0][0]
    plt.title(tmp)
    plt.show()
    if int(obs_y[obs_predicted.argmax()])==np.where(label==1)[0][0]:
        sum=sum+1
    count = count + 1
    if count==100:
        break

print(sum/100)