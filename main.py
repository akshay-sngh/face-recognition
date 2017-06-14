import cv2
from PyQt4 import QtGui, QtCore
import sqlite3
from datetime import datetime

#cascase classifiers for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
myList = ['Unknown','Akshay','Abhishek','Audi','Viji']
finalId =0


def getInfo(id):
    '''This function retrieves the profile from the database'''
    #get connection to database
    conn = sqlite3.connect('FaceBase.db')

    #running the sql query
    query = 'SELECT * FROM Employee WHERE Id='+str(id)

    rows = conn.execute(query)
    info = None
    for row in rows:
        info = row
    conn.close()

    return info

def recordDate(id):

    '''This function stores the current time in the database for the persond with ID = id'''

    # if person is unknown then don't record any Entry
    if id is 0:
        return

    #get today's date using datetime module from python
    dateString = str(datetime.now().date())

    conn = sqlite3.connect('FaceBase.db')
    print ('dateString is ',dateString)

    #Update the date entry in the database
    query = (" UPDATE Employee SET Entry='%s' WHERE Id=%d " % (dateString,id))
    print (query)
    rows = conn.execute(query)

    conn.commit()

    conn.close()
    # print rows


#invoke recognizer objects
recEigen = cv2.createEigenFaceRecognizer()
recFisher = cv2.createFisherFaceRecognizer()
recLBPH = cv2.createLBPHFaceRecognizer()


def resizeImage(image,size=(50,50)):
    '''this function resizes the image to 50x50 for Eigen and Fisher Reconginizers'''
    if image.shape < size:
        image = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image,size,interpolation = cv2.INTER_CUBIC)
    return image

def cutImage(image,x,y,w,h):
    '''crop the image so that only the face retains'''
    w_rm = int(0.2*w/2)
    image = image[y:y+h, x+w_rm : x+ w- w_rm]
    return image

def normalizePixels(image):
    '''normalizes the pixel density to adjust contrast the image'''
    image = cv2.equalizeHist(image)
    return image
#loading the data that used for training purposes
recEigen.load('recognizerData/EigenData.yml')
recFisher.load('recognizerData/FisherData.yml')
recLBPH.load('recognizerData/LBPHData.yml')
#font style for the prediction text
font = cv2.cv.InitFont(cv2.FONT_HERSHEY_PLAIN,1.7,1,0,1,30)

class Capture():
    def __init__(self):
        self.capturing = False
        self.c = cv2.VideoCapture(0)
        ret = self.c.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,600);
        ret = self.c.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,600);

    def startCapture(self):
        print ("pressed start")
        self.capturing = True
        cap = self.c
        while(self.capturing):
            ret, img = cap.read()
            #converting image into grayscale
            gray_scale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # img = cv2.flip(img,1)
            #detect faces from the greyscale image
            faces = faceDetect.detectMultiScale(gray_scale,1.3,5)
            #use the coordinates of the ROI to draw a rectangle around it
            global finalId
            for(x,y,w,h) in faces:
                #draw a rectangle around the face
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                #converting into grayscale
                gray = cutImage(gray_scale,x,y,w,h)

                #normalizing pixels
                gray = normalizePixels(gray)


                #resizing the image
                gray = resizeImage(gray)

                myId = 0

                id1,conf1 = recLBPH.predict(gray)
                print ('LBPH says',myList[id1], 'recognised with a confidence of ' , conf1)
                #cv2.cv.PutText(cv2.cv.fromarray(img),'LPBH::'+myList[id1],(x,y+h+25),font,(0,255,0),)


                id2,conf2 = recFisher.predict(gray)
                print ('Fisher says',myList[id2], 'recognised with a confidence of ' , conf2)
                #cv2.cv.PutText(cv2.cv.fromarray(img),'Fisher::'+myList[id2],(x,y+h+45),font,(0,255,0),)
                #if any two IDs match then identify the person
                if id1 is id2:
                    myId = id2




                id3,conf3 = recEigen.predict(gray)
                print ('Eigen says',myList[id3], 'recognised with a confidence of ' , conf3)
                #cv2.cv.PutText(cv2.cv.fromarray(img),'Eigen::'+myList[id3],(x,y+h+65),font,(0,255,0),)
                if id1 is id3:
                    myId = id3
                elif id2 is id3:
                    myId = id3


                #print the name of the person with atleast 2 out of 3 votes
                info = getInfo(myId)
                # cv2.cv.PutText(cv2.cv.fromarray(img),'Name::'+str(info[1]),(x,y+h+25),font,(0,255,0),)
                # cv2.cv.PutText(cv2.cv.fromarray(img),'CorpID::'+str(info[2]),(x,y+h+50),font,(0,255,0),)
                # cv2.cv.PutText(cv2.cv.fromarray(img),'Business Unit::'+str(info[3]),(x,y+h+75),font,(0,255,0),)

                #printing the name on the left bottom corner of the screen
                cv2.cv.PutText(cv2.cv.fromarray(img),'Name::'+str(info[1]),(3,400),font,(0,255,0),)
                cv2.cv.PutText(cv2.cv.fromarray(img),'CorpID::'+str(info[2]),(3,420),font,(0,255,0),)
                cv2.cv.PutText(cv2.cv.fromarray(img),'Business Unit::'+str(info[3]),(3,440),font,(0,255,0),)

                finalId = myId


                print ('____________________________________________')
                print


            cv2.namedWindow("Capture",cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Capture", img)
            cv2.waitKey(5)
        cv2.destroyAllWindows()


    def setTime(self):
        print ('ID received is !', finalId)
        #make a new message box

        recordDate(finalId)


    def endCapture(self):
        print ("pressed End")
        global finalId
        finalId = 0
        self.capturing = False


    def quitCapture(self):
        self.capturing = False
        print ("pressed Quit")
        cap = self.c
        cv2.destroyAllWindows()
        cap.release()
        QtCore.QCoreApplication.quit()


class Window(QtGui.QWidget):
    def __init__(self):

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.capture = Capture()
        self.start_button = QtGui.QPushButton('Open',self)
        self.start_button.clicked.connect(self.capture.startCapture)

        self.end_button = QtGui.QPushButton('Close',self)
        self.end_button.clicked.connect(self.capture.endCapture)

        self.record_button = QtGui.QPushButton('Record Time',self)
        self.record_button.clicked.connect(self.capture.setTime)

        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(self.capture.quitCapture)

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.record_button)
        vbox.addWidget(self.quit_button)

        self.setLayout(vbox)
        self.setGeometry(100,100,250,200)
        self.show()


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
