
import numpy as np
import cv2
import imutils
import math
from imutils.video import VideoStream
from fin_directkeys import PressKey, W, A,S,D, Shift, ReleaseKey

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")
offset = 20
imgSize = 300
labels = ["FIRE", "Jump"]
currentKey = list()
cam = VideoStream(src=0).start()
spaceCounter = 1
while True:

    key = False
    try:
        img = cam.read()
        img = np.flip(img,axis=1)
        img = imutils.resize(img, width=840)
        img = imutils.resize(img, height=680)
        imgOutput = img.copy()

        hsv = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2HSV)
        value = (11, 11)
        blurred = cv2.GaussianBlur(hsv, value,0)
        colourLower = np.array([95, 69, 131])
        colourUpper = np.array([180,255,255])

        height = imgOutput.shape[0]
        width = imgOutput.shape[1]

        mask = cv2.inRange(blurred, colourLower, colourUpper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        upContour = mask[height//4:3*(height//4),0:width]

        cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts_up = imutils.grab_contours(cnts_up)

        if len(cnts_up) > 0:
            
            c = max(cnts_up, key=cv2.contourArea)
            M = cv2.moments(c)
            cX = int(M["m10"]/(M["m00"]+0.000001))

            if cX < (width//4 + 10):
                PressKey(A)
                key = True
                currentKey.append(A)
            elif cX > ((3*(width//4 ))-10):
                PressKey(D)
                key = True
                currentKey.append(D)
    ######################################################################################################################################
        # img = np.flip(img,axis=1)
        hands, img = detector.findHands(imgOutput)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(prediction, index)
            if index==1:
                key = True
                # if spaceCounter==1:
                #     PressKey(W)
                # elif spaceCounter==3:
                #     ReleaseKey(W)
                # elif spaceCounter==6:
                #     spaceCounter=0
                # spaceCounter+=1
                # currentKey.append(W)
                PressKey(W)
                key = True
                currentKey.append(W)
            
            elif index==0:
                PressKey(Shift)
                key = True
                currentKey.append(Shift)
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),(x - offset+200, y - offset-50+50), (0, 137, 112), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (53, 238, 223 ), 2)
            # cv2.rectangle(imgOutput, (x-offset, y-offset),(x + w+offset, y + h+offset), (0, 255, 0), 4)

    ######################################################################################################################################
        
        # if len(cnts_down) > 0:hhh
        #     PressKey(Space)
        #     key = True
        #     currentKey.append(Space)
        
        cv2.rectangle(imgOutput,(0,height//4),(width//4,3*(height//4) ),(0, 137, 112),2)

        cv2.rectangle(imgOutput,(3*(width//4),height//4),(width-2,3*(height//4) ),(0, 137, 112),2)

        # img = cv2.rectangle(img,(2*(width//5),3*(height//4)),(3*width//5,height),(0,255,0),1)
        # cv2.putText(img,'JUMP',(2*(width//5) + 20,height-10),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))
        
        cv2.imshow("Controls", imgOutput)

        if not key and len(currentKey) != 0:
            for current in currentKey:
                ReleaseKey(current)
            currentKey = list()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except Exception as e:
        print(e)
 
cv2.destroyAllWindows()