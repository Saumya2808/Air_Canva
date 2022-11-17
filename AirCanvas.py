import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 10
eraserThickness = 100

folderPath = 'Virtual Painter'
myImg = os.listdir(folderPath)
print(myImg)

overlayList = []
for img in myImg:
    image = cv2.imread(f'{folderPath}/{img}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.65, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # 1. Import Image

    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks

    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        # print(lm_list)
        # tip of index and middle finger
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        # 3. Check which finger is up

        fingers = detector.fingersUp()
        print(fingers)
        # print(fingers)

        # 4. If Selection mode, then two fingers are up

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15), (255, 0, 255), cv2.FILLED)
            print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing mode, then indexfinger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            # print('Drawing Mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header

    # Transparency between two windows
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow('Canvas', imgCanvas)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

