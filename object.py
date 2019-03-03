import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import glob
from inverse_kinetics import inverse_kinematics
from alexnet import AlexNet
import tensorflow as tf
import numpy as np
from collections import Counter
import time


options ={
    'model' : 'cfg/tiny-yolo-voc-1c.cfg',
    'load' : 12400,
    'threshold': 0.06,
    'gpu' : 0.6
    }

tfnet = TFNet(options)
inverse = inverse_kinematics("/dev/rfcomm0",38400,1)

#cap = cv2.VideoCapture(1)
count = 0
ready = 0
images = []

diseases = ["Tomato___Bacterial_spot"
"Tomato___Early_blight",
"Tomato___Late_blight",
"Tomato___Leaf_Mold",
"Tomato___Septoria_leaf_spot",
"Tomato___Spider_mites Two-spotted_spider_mite",
"Tomato___Target_Spot",
"Tomato___Tomato_mosaic_virus",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___healthy"]

X = tf.placeholder(tf.float32, [None, 227, 227, 3])
Y = tf.placeholder(tf.float32 ,[None ,10])
keep_prob = tf.placeholder(tf.float32)
model = AlexNet(X, keep_prob, 10, [])
score = model.fc8
compare = tf.argmax(score,1)
saver = tf.train.Saver() 
f= open("plant.txt","w+")
coordinate_for_camera = [[20 ,15,25] , [20, 15 ,20], [ 20 ,15 ,15]]
ready = 0
number_of_plants = 4
co = 1

a = -39
b =42
a = str(a) + 'p'
b = str(b) + 'q'
d = 50

distance = str(d)+ "d"
inverse.kin.write(distance.encode())
time.sleep(6)
print("Distance traveled completed===>")
inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(6)
print("Arm lifting completed")



cap = cv2.VideoCapture("http://192.168.43.1:8080/video?x.mjpeg")
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
ret ,frame  = cap.read(cv2.IMREAD_COLOR)
print(frame.shape)
if ret == True:
	result= tfnet.return_predict(frame)
	print(result)
	if not result == []:	
		for res in result:
			tl = (res['topleft']['x'], res['topleft']['y'])
			br = (res['bottomright']['x'], res['bottomright']['y'])
			label = res['label']
			confidence = res['confidence']
			croped = frame[res['topleft']['y']:res['bottomright']['y'], res['topleft']['x']:res['bottomright']['x'],:]
			croped = cv2.resize(croped ,(227, 227))
			croped = croped /225
			images.append(croped)
			frame = cv2.rectangle(frame, tl, br, (255, 0 , 0), 5)
			frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)

		num_images = np.array(images)	
		with tf.Session() as sess:
			saver.restore(sess, "/home/sansii/Desktop/Agro_doctor/checkpoint/model.ckpt")
			predict = sess.run(compare , {X:num_images,keep_prob :1})
		
		max = Counter(predict).most_common(1)[0][0]

		print(predict)
		print(max)
		print("Predict====>"+str(predict))
		string = "Diseases===>"+str(diseases[max])+"\n"
		print(string)
		f.write(string)

	
		

	cv2.imshow('image',frame)
	cv2.imwrite('clicked_picture'+str(1)+".jpg",frame)
	#np.save('Diseases.txt',predict)
	K = cv2.waitKey(0)
	if K == 27:
		sess.close()
		#cap.release()
		cv2.destroyAllWindows()	


inverse.kin.write(b'o')
time.sleep(3)
inverse.kin.write(b'f')
time.sleep(1)
print("Pump spraying completed")
a = -10
b =50
a = str(a) + 'p'
b = str(b) + 'q'

inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(3)
print("Arm in next position completed")

####1st series
print("2nd plant started")

a = -39
b =42
a = str(a) + 'p'
b = str(b) + 'q'
d = 100
distance = str(d)+ "d"
inverse.kin.write(distance.encode())
time.sleep(4)
print("Distance 2nd completed")
inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(3)
print("Arm lifting 2nd completed")

cap = cv2.VideoCapture("http://192.168.43.1:8080/video?x.mjpeg")
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
ret ,frame  = cap.read(cv2.IMREAD_COLOR)
print(frame.shape)
ret = True
if ret == True:
	result= tfnet.return_predict(frame)
	print(result)
	if not result == []:	
		for res in result:
			tl = (res['topleft']['x'], res['topleft']['y'])
			br = (res['bottomright']['x'], res['bottomright']['y'])
			label = res['label']
			confidence = res['confidence']
			croped = frame[res['topleft']['y']:res['bottomright']['y'], res['topleft']['x']:res['bottomright']['x'],:]
			croped = cv2.resize(croped ,(227, 227))
			croped = croped /225
			images.append(croped)
			frame = cv2.rectangle(frame, tl, br, (255, 0 , 0), 5)
			frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)

		num_images = np.array(images)	
		with tf.Session() as sess:
			saver.restore(sess, "/home/sansii/Desktop/Agro_doctor/checkpoint/model.ckpt")
			predict = sess.run(compare , {X:num_images,keep_prob :1})
		
		max = Counter(predict).most_common(1)[0][0]

		print(predict)
		print(max)
		print("Predict====>"+str(predict))
		string = "Diseases===>"+str(diseases[max])+"\n"
		print(string)
		f.write(string)

	
		

	cv2.imshow('image',frame)
	cv2.imwrite('clicked_picture'+str(1)+".jpg",frame)
	#np.save('Diseases.txt',predict)
	K = cv2.waitKey(0)
	if K == 27:
		sess.close()
		#cap.release()
		cv2.destroyAllWindows()
print("Image_capture_completed")
inverse.kin.write(b'o')
time.sleep(3)
inverse.kin.write(b'f')
time.sleep(1)
print("spray completed")

a = -10
b =50
a = str(a) + 'p'
b = str(b) + 'q'

time.sleep(2)
inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(2)
print("Arm 2nd time lifting completed")
####2nd series

# elif ready == 0:
# 	distance = str(50 )+ "d"
# 	inverse.kin.write(distance.encode())
# 	count += 1
# 	ready = 3

# ready = inverse.kin.read()
# print(ready)
"""
a = -39
b =42
a = str(a) + 'p'
b = str(b) + 'q'
d = 150
distance = str(d)+ "d"
inverse.kin.write(distance.encode())
time.sleep(4)
inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(4)

# ret ,frame  = cap.read(cv2.IMREAD_COLOR)


# print(frame.shape)
#ret = True
# if ret == True:
# 	result= tfnet.return_predict(frame)d = d + i*50
# 	print(result)
# 	if not result == []:	
# 		for res in result:
# 			tl = (res['topleft']['x'], res['topleft']['y'])
# 			br = (res['bottomright']['x'], res['bottomright']['y'])
# 			label = res['label']
# 			confidence = res['confidence']
# 			croped = frame[res['topleft']['y']:res['bottomright']['y'], res['topleft']['x']:res['bottomright']['x'],:]
# 			croped = cv2.resize(croped ,(227, 227))
# 			croped = croped /225
# 			images.append(croped)
# 			frame = cv2.rectangle(frame, tl, br, (255, 0 , 0), 5)
# 			frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_DUPLEX,1, (0,0,0),2)

# 		num_images = np.array(images)	
# 		with tf.Session() as sess:
# 			saver.restore(sess, "/home/sansii/Desktop/Agro_doctor/checkpoint/model.ckpt")
# 			predict = sess.run(compare , {X:num_images,keep_prob :1})
		
# 		max = Counter(predict).most_common(1)[0][0]

# 		print(predict)
# 		print(max)
		#print("Predict====>"+str(predict))
		# string = "Diseases===>"+str(diseases[max])+"\n"
		# f.write(string)

	# else:
	# 	inverse.calculate_angle(coordinate_for_camera[i])

	
		

	# cv2.imshow('image',frame)
	# #np.save('Diseases.txt',predict)
	# K = cv2.waitKey(10)
	# if K == 27:
	# 	sess.close()
	# 	#cap.release()
	# 	cv2.destroyAllWindows()
	# 	break	

inverse.kin.write(b'o')
time.sleep(2)
inverse.kin.write(b'f')
time.sleep(1)

a = -10
b =50
a = str(a) + 'p'
b = str(b) + 'q'

time.sleep(4)
inverse.kin.write(a.encode())
inverse.kin.write(b.encode())
time.sleep(2)
###end


# elif ready == 0:
# 	distance = str(50 )+ "d"
# 	inverse.kin.write(distance.encode())
# 	count += 1
# 	ready = 3

# ready = inverse.kin.read()
# print(ready)
"""
print("Ready for back ")
d = 10
distance = str(d)+ "d"
inverse.kin.write(distance.encode())


