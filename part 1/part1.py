import pandas as pd
import numpy as np
import face_recognition
import cv2
import time

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



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
	
		#print("----yes-----")
		#print(face_locations)
		
		frame = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
		#print(len(image), len(image[0]), len(image[0][0]))
		#print(face_locations)
		#cv2.imwrite('face/'+img[i], frame)
		
		#break
		
		image_array.append((np.array(cv2.resize(frame, (200, 200))).flatten())/255)
		label_array.append(player_list.index(player[i]))

	#image_array.append(np.array((cv2.resize(face_recognition.load_image_file('all_images/'+img[i]), (224, 224))).flatten())/255)








x_train, x_test, y_train, y_test = train_test_split(image_array, label_array, test_size = 0.3)







clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)




y_tra = clf.predict(x_train)
y_tes = clf.predict(x_test)





print("-------- Random Forest --------")
print()

print("Confusion matrix:")
print()
print("Training accuracy:", metrics.accuracy_score(y_tra, y_train)*100, "%")
print()

print(metrics.confusion_matrix(y_tra, y_train))
print()
print()

print("Confusion matrix:")
print()
print("Testing accuracy:", metrics.accuracy_score(y_tes, y_test)*100, "%")
print()

print(metrics.confusion_matrix(y_tes, y_test))


print()
print()









clf = LogisticRegression()
clf = clf.fit(x_train, y_train)




y_tra = clf.predict(x_train)
y_tes = clf.predict(x_test)



print("-------- Logistic Regression --------")
print()

print("Training accuracy:", metrics.accuracy_score(y_tra, y_train)*100, "%")
print()

print("Confusion matrix:")
print()

print(metrics.confusion_matrix(y_tra, y_train))
print()
print()

print("Testing accuracy:", metrics.accuracy_score(y_tes, y_test)*100, "%")
print()

print("Confusion matrix:")
print()
print(metrics.confusion_matrix(y_tes, y_test))

print()
print()









clf = SVC(kernel = 'linear')
clf = clf.fit(x_train, y_train)




y_tra = clf.predict(x_train)
y_tes = clf.predict(x_test)



print("-------- SVM (Kernel = linear) --------")
print()

print("Training accuracy:", metrics.accuracy_score(y_tra, y_train)*100, "%")
print()

print("Confusion matrix:")
print()
print(metrics.confusion_matrix(y_tra, y_train))
print()
print()

print("Testing accuracy:", metrics.accuracy_score(y_tes, y_test)*100, "%")
print()

print("Confusion matrix:")
print()
print(metrics.confusion_matrix(y_tes, y_test))









clf = SVC(kernel = 'poly')
clf = clf.fit(x_train, y_train)




y_tra = clf.predict(x_train)
y_tes = clf.predict(x_test)



print("-------- SVM (Kernel = polynomial) --------")
print()

print("Training accuracy:", metrics.accuracy_score(y_tra, y_train)*100, "%")
print()

print("Confusion matrix:")
print()
print(metrics.confusion_matrix(y_tra, y_train))
print()
print()

print("Testing accuracy:", metrics.accuracy_score(y_tes, y_test)*100, "%")
print()

print("Confusion matrix:")
print()
print(metrics.confusion_matrix(y_tes, y_test))
