import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(1)
cap.set(3,320)
cap.set(4,240)

top,right,bottom,left= 55,95,230,300 

while True:


    lowerBound_green=np.array([0,0,0])
    upperBound_green=np.array([91,120,255])

    
    lower_red = np.array([117,0,0])
    upper_red = np.array([180,255,255])

    kernelOpen = np.ones((5,5))
    kernelClose = np.ones((20,20))
    kernelOpen1=np.ones((5,5))
    kernelClose1=np.ones((20,20))


    
    grabbed,frame=cap.read()
    frame=imutils.resize(frame,width=400)
    frame=cv2.flip(frame,1)
    clone=frame.copy()

    #(height,width)=frame.shape[:2]
    roi = frame[top:bottom, right:left]


    
    
    imgHSV= cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

    mask=cv2.inRange(imgHSV,lowerBound_green,upperBound_green)
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose

    _,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgHSV,conts,-1,(255,255,0),2)


    mask1=cv2.inRange(imgHSV,lower_red,upper_red)
    maskOpen1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernelOpen1)
    maskClose1=cv2.morphologyEx(maskOpen1,cv2.MORPH_CLOSE,kernelClose1)
    maskFinal1=maskClose1
    
    _,contours2,h = cv2.findContours(maskFinal1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgHSV,contours2,-1,(0,255,255),3)



    
    

    cv2.rectangle(clone, (left,top), (right,bottom), (0,255,0),2)

    

    # display the frame with segmented hand
    
    cv2.imshow("Apple2",imgHSV)
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

# free up memory
cap.release()
cv2.destroyAllWindows()
