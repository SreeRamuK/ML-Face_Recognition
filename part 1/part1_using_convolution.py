import pandas as pd
import numpy as np
import face_recognition
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools
import keras
import tensorflow as tf



from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.utils import np_utils
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout, Conv2D, MaxPool2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical





features = ['sl_no', 'image_id', 'player_name']

data = pd.read_csv("players.csv", header = None, names = features)


img_id = data.image_id
player_id = data.player_name


img = []
player = []

for i in range(1, len(img_id)):

	img.append(img_id[i])

for i in range(1, len(player_id)):

	player.append(player_id[i])



player_list = []

for i in range(len(player)):

	if(player[i] not in player_list):
	
		player_list.append(player[i])





label_array = []


for i in range(len(player)):

	label_array.append(player_list.index(player[i]))






label_array = []
image_array = []


for i in range(len(img)):

	image = face_recognition.load_image_file('all_images/'+img[i])
	face_locations = face_recognition.face_locations(image)
	
	num = len(face_locations)
	
	if(num == 1):
		
		frame = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
		
		image_array.append((np.array(cv2.resize(frame, (20, 20))))/255)
		label_array.append(player_list.index(player[i]))








x_train, x_test, y_train, y_test = train_test_split(image_array, label_array, test_size = 0.3)





Y_train = to_categorical(y_train, 15)
Y_test = to_categorical(y_test, 15)

x_train = np.array(x_train)
Y_train = np.array(Y_train)

x_test = np.array(x_test)
Y_test = np.array(Y_test)










n_classes = 15


Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)




model = Sequential()


model.add(Conv2D(50, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu', input_shape = (20, 20, 3)))
model.add(Conv2D(75, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(250, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(15, activation = 'softmax'))


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x_train, Y_train, batch_size = 10, epochs = 100, validation_data = (x_test, Y_test))







y_tra = model.predict(x_train)

y_pred = []

for i in range(len(Y_train)):

	max1 = 0
	m1 = -1

	for j in range(15):
		
		if(y_tra[i][j] > max1):
		
			max1 = y_tra[i][j]
			m1 = j
	
	y_pred.append(m1)


print()
print()
print("Training accuracy:", metrics.accuracy_score(y_train, y_pred)*100, "%")
print()
print(metrics.confusion_matrix(y_train, y_pred))
print()
print()


y_tra = model.predict(x_test)

y_pred = []

for i in range(len(Y_test)):

	max1 = 0
	m1 = -1

	for j in range(15):
		
		if(y_tra[i][j] > max1):
		
			max1 = y_tra[i][j]
			m1 = j
	
	y_pred.append(m1)

	
print("Testing accuracy:", metrics.accuracy_score(y_test, y_pred)*100, "%")
print()
print(metrics.confusion_matrix(y_test, y_pred))
