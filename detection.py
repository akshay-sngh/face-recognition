import numpy as np
import cv2
import time

def detectFace():
	faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #calling classifier and storing in a variable)

	cam = cv2.VideoCapture(0) #capture images from the webcam captureID = 0

	#capture frames one by one and detect the faces in the window
	while True:
		ret,img = cam.read()	#returns status variable and captured colored image
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)	#convert into grayscale

		#list that stores faces

		faces = faceDetect.detectMultiScale(gray,1.3,5);	#detects all the faces and returns the coordinate of the faces

		# for multiple faces and drawing rectangles
		for(x,y,w,h) in faces:
			#draw a rectangle on colored image
			cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,256),2) #BGR

		#displaying the image in the window
		cv2.imshow("Face",img);

		#openCV works only with the wait command
		if cv2.waitKey(50) == ord('q'):	#quits if q is pressed
			break


	#releasing the camera for operation
	cam.release()
	cv2.destroyAllWindows()
detectFace()
