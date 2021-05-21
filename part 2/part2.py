import pygame
import pygame.camera
import time
import pandas as pd
import numpy as np
import face_recognition
import cv2


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression





print()
print("Image will get captured in 10 seconds")

time.sleep(10)


pygame.camera.init()
pygame.camera.list_cameras()

cam = pygame.camera.Camera("/dev/video0",(3840,2160))
cam.start()

img = cam.get_image()

pygame.image.save(img,"test_image.jpg.jpg")

print()
print("-------- Image captured --------")

print()
print()









print("-------- Loading Data --------")
print()

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





print("-------- Data Pre processing started --------")
print()

label_array = []
image_array = []


for i in range(len(img)):

	image = face_recognition.load_image_file('all_images/'+img[i])
	face_locations = face_recognition.face_locations(image)
	
	num = len(face_locations)
	
	if(num == 1):
	
		#print("----yes-----")
		#print(face_locations)
		
		frame = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
		#print(len(image), len(image[0]), len(image[0][0]))
		#print(face_locations)
		#cv2.imwrite('face/'+img[i], frame)
		
		#break
		
		image_array.append((np.array(cv2.resize(frame, (200, 200))).flatten())/255)
		label_array.append(player_list.index(player[i]))










print("-------- Starting to predict captured image --------")
print()

time.sleep(5)


test_img = []

image = face_recognition.load_image_file('test_image.jpg')
face_locations = face_recognition.face_locations(image)

num = len(face_locations)

if(num == 0):

	print("Error: Captured image is not clear or Captured image has no face")
	exit()

if(num > 1):

	print("Error: Captured image has multiple faces")
	exit()
	


elif(num == 1):

	frame = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
	test_img.append((np.array(cv2.resize(frame, (200, 200))).flatten())/255)







print("-------- Designing Random forest model --------")

clf = RandomForestClassifier()
clf = clf.fit(image_array, label_array)

print("-------- Random forest model completed --------")
print()

y1 = clf.predict(test_img)




print("-------- Designing Logistic Regression model --------")

clf = LogisticRegression()
clf = clf.fit(image_array, label_array)

print("-------- Logistic Regression model completed --------")
print()

y2 = clf.predict(test_img)




print("-------- Designing Linear SVM model --------")

clf = SVC(kernel = 'linear')
clf = clf.fit(image_array, label_array)

print("-------- Linear SVM model completed --------")
print()

y3 = clf.predict(test_img)




print("-------- Designing Polynomial SVM model --------")

clf = SVC(kernel = 'poly')
clf = clf.fit(image_array, label_array)

print("-------- Polynomial SVM model completed --------")
print()

y4 = clf.predict(test_img)





print()
print("Random forest model decision:", player_list[y1[0]])
print("Logistic Regression model decision:", player_list[y2[0]])
print("Linear SVM model decision:", player_list[y3[0]])
print("Polynomial SVM model decision", player_list[y4[0]])




imp_value = [1.5, 2.5, 2.5, 2]


value = []


for i in range(15):

	value.append(0)


value[y1[0]] = value[y1[0]] + imp_value[0]
value[y2[0]] = value[y2[0]] + imp_value[1]
value[y3[0]] = value[y3[0]] + imp_value[2]
value[y4[0]] = value[y4[0]] + imp_value[3]



if(y1[0] != y2[0] and y1[0] != y3[0] and y1[0] != y4[0] and y2[0] != y3[0] and y2[0] != y4[0] and y3[0] != y4[0]):

	print()
	print("Warning: Each model is decision is different!! Please check the photo again it might not belong to anyone....")


else:

	print()
	print("Most probably it might be", player_list[value.index(max(value))])
