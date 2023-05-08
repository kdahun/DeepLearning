from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os, glob

import matplotlib.pyplot as plt

batchsize =5
path = os.path.dirname(os.path.abspath(__file__))

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 30
								   , width_shift_range=0.2, height_shift_range=0.2
																		, shear_range = 0.2, zoom_range = 0.2
																		, horizontal_flip = False, fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(path+'/train', target_size=(128,128)
                                                    , batch_size = batchsize
                                                    , class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(path+'/test', target_size=(128,128)
                                                  , batch_size = batchsize
                                                  , class_mode= 'categorical')

n_img = train_generator.n
imgs, labels = [],[]
for i in range(n_img):
	img,lable = train_generator.next()
	imgs.extend(img)
	labels.extend(lable)

imgs=np.asarray(imgs)
labels = np.asarray(labels)

plt.figure(figsize = (10,10))

for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(imgs[i],cmap=plt.cm.binary)

plt.show()