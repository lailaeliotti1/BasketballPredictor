import math

import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

#Initialize the video
cap = cv2.VideoCapture('Videos/vid (4).mp4')

#Create the color finder object
myColorFinder = ColorFinder(False)
hsvValues = {'hmin': 0, 'smin': 105, 'vmin': 0, 'hmax': 15, 'smax': 255, 'vmax': 255}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]



while True:
    #Grab the image
    success, img = cap.read()
    img = img[0:900, :]

    #Find the color of the ball
    imgColor, mask = myColorFinder.update(img, hsvValues)

    #Find the location
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    #Display each point of ball
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        # find coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)


        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            position = (posX, posY)
            cv2.circle(imgContours, position, 10, (0,255,0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, position, position, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, position, (posListX[i-1], posListY[i-1]), (0,255, 0), 2)

    # Polynomial Regression y = Ax^2 + Bx + C
        for x in xList:
            y = int(A*x**2 + B*x + C)
            cv2.circle(imgContours, (x,y), 2, (255, 0, 255), cv2.FILLED)

        # Prediction
        # X values 330 to 430  Y 590
        a = A
        b = B
        c = C - 590

        x = int((-b - math.sqrt(b**2 - (4*a*c)))/(2*a))
        print(x)
        if 330<x<430:
            cvzone.putTextRect(imgContours, "BASKET", (50,100), scale= 7,
                               thickness=5, colorR=(0,200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "NO BASKET", (50, 100), scale=7,
                               thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(100)




