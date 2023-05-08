import cv2
import os,glob
import numpy
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np


image_size =(128,128)
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
        obs_img=cv2.imread(j)
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

new_obs_model=keras.models.load_model('obs_model.h5')
print(obs_list)

obs_y=['0','1']

# for img in obs_img_file:
#     frame=cv2.imread(img)
#     frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     frame=cv2.resize(frame,(128,128))
#     predicted_result= new_obs_model.predict(np.array([frame]))
#     obs_predicted = predicted_result[0]
#     for i,obsClass in enumerate(obs_predicted):
#         print(obs_Ylable[i],'=',int(obsClass*100))
#     print('Predicted Result = ',obs_y[obs_predicted.argmax()])
#     plt.imshow(frame)
#     tmp="Prediction"+obs_y[obs_predicted.argmax()]
#     plt.title(tmp)
#     plt.show()