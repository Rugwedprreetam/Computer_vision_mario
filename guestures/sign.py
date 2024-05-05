import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import keyboard
from fin_directkeys import PressKey, A, D, W, Shift,ReleaseKey
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300


currentKey = list() 
currentKey.append(D)
counter = 0
counter_flag = 1
spaceCounter = 0
def release(cur):
    for current in cur:
        ReleaseKey(current)
    cur = list()
while True:
    key = False
    try:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
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
            cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                        (x - offset+190, y - offset-50+50), (255, 0, 255), cv2.FILLED)
            # cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset),
                        (x + w+offset, y + h+offset), (255, 0, 255), 4)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
            if counter_flag==index and counter>=2:
                if index==3:
                    keyboard.press_and_release("FIRE")
                    PressKey(Shift)
                    key = True
                    currentKey.append(Shift)
                if index==2:
                    # keyboard.press_and_release("JUMP")
                    key = True
                    if spaceCounter==1:
                        PressKey(W)
                    elif spaceCounter==3:
                        ReleaseKey(W)
                    elif spaceCounter==6:
                        spaceCounter=0
                    spaceCounter+=1
                if index==0:
                    # keyboard.press_and_release("LEFT")
                    # release(currentKey)
                    release(currentKey)
                    PressKey(A)
                    key = True
                    currentKey.append(A)
                if index==1:
                    # keyboard.press_and_release("RIGHT")
                    release(currentKey)
                    PressKey(D)
                    key = True
                    currentKey.append(D)
                print("case1"+str(index)+" "+str(counter_flag)+" "+str(counter))
            elif counter_flag==index and counter<2:
                counter+=1
                print("case2"+str(index)+" "+str(counter_flag)+" "+str(counter))
            else:
                counter=0
                counter_flag=index
                print("case3"+str(index)+" "+str(counter_flag)+" "+str(counter))
        imgOutput = cv2.resize(imgOutput,(1660,750))
        cv2.imshow("Image", imgOutput)
        key1 = cv2.waitKey(10)
        if key1 == ord("q"):
            break
        # if not key and len(currentKey) != 0:
        #     release()
    except Exception as e:
        print(e)