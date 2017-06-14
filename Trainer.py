
import os
import cv2
import numpy as np

recognizerEigen = cv2.createEigenFaceRecognizer()
recognizerFisher = cv2.createFisherFaceRecognizer()
recognizerLBPH = cv2.createLBPHFaceRecognizer()

path = 'dataSet'
newpath = 'newSet'

#for E(igen and Fisher
def getImagesWithID(path,flag):
	'''get images and their corresponding IDs'''

	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

	#join appends filename to the path

	faces = []
	Ids = []
	for imagePath in imagePaths:

		faceImg = cv2.imread(imagePath)
		#converting into greyscale
		faceImg = cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
		#the above two scripts can also be done this way
		#faceImg = Image.open(imagePath).convert('L')

		#converting to numpy array so the openCV can work on it
		faceNp = np.array(faceImg,'uint8')

		#retrieve ID of the user from the image path
		ID = int(os.path.split(imagePath)[-1].split('.')[1])

		#storing them in the lists
		faces.append(faceNp)
		Ids.append(ID)

		if flag is 1:
			cv2.imshow('training',faceNp)
			cv2.waitKey(300)
			print ID

	return np.array(Ids),faces

'''main program starts here'''
#get the images and their corresponding Ids
Ids,faces = getImagesWithID(path,0)

recognizerEigen.train(faces,Ids)
recognizerEigen.save('recognizerData/EigenData.yml')

recognizerFisher.train(faces,Ids)
recognizerFisher.save('recognizerData/FisherData.yml')

Ids,faces = getImagesWithID(path,1)

recognizerLBPH.train(faces,Ids)
recognizerLBPH.save('recognizerData/LBPHData.yml')