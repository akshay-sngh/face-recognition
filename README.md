# face-recognition

## About the app
Opencv face recognition using python by loading data from sqlite3 database and controlling the application using PyQt4 GUI.

change the database to link your new faces in it
for any queries or troubleshooting - akshaysingh000@gmail.com :)


requires:
1. Numpy
2. OpenCV version 2.4.x to 3.0.0, newer versions required additonal codes for training the data
3. PyQt4

# How to run:
Run **main.py**

## Training additional set of images:
1. Create folder called trainSet and put your new faces and name the image file witgh their respective IDs (like 1.0,1.1 and 2.0,etc.)
2. Run testCreator.py so that your faces are detected
3. Run Trainer.py so new training data is saved on the folder 'recognizerData'
4. Run main.py
