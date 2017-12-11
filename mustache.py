#Robotic Systems - 

import cv2   
import numpy as np

#Set-up path for detection
basePath = "C:/Python27/opencv/sources/data/haarcascades/"
 
facePath = basePath + "haarcascade_frontalface_default.xml"
nosePath = basePath + "haarcascade_mcs_nose.xml"
 
faceCascade = cv2.CascadeClassifier(facePath)
noseCascade = cv2.CascadeClassifier(nosePath)
 
mustacheImage = cv2.imread('mustache.png',-1)
 
# Make the mask for the image 
mask = mustacheImage[:,:,3]
 
# Make the Inverted mask
invertedMask = cv2.bitwise_not(mask)
 
# Get the original size of image
mustacheImage = mustacheImage[:,:,0:3]
origMustacheHeight, origMustacheWidth = mustacheImage.shape[:2]
 
# Access Webcam input 
webcam = cv2.VideoCapture(0)

#Create videoWriter object
fourcc = cv2.cv.CV_FOURCC('i','Y','U','V')
out = cv2.VideoWriter('output.avi',fourcc, 5.0, (640,480))
 
while True:
    # Get video feed
    ret, frame = webcam.read()
 
    # Turn video feed into gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # Detect faces in video feed
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
 
   # Iterate over each face found
    for (x_Coord, y_Coord, width, height) in faces:

        # draw box around face
        face = cv2.rectangle(frame,(x_Coord,y_Coord),(x_Coord+width,y_Coord+height),(255,0,0),2)

        # Set region of image 
        roiGray = gray[y_Coord:y_Coord+height, x_Coord:x_Coord+width]
        roiColor = frame[y_Coord:y_Coord+height, x_Coord:x_Coord+width]
 
        # Detect a nose
        nose = noseCascade.detectMultiScale(roiGray)
 
        for (nose_x_Coord,nose_y_Coord,nose_width,nose_height) in nose:
            
            # Draw box around nose
            cv2.rectangle(roiColor,(nose_x_Coord,nose_y_Coord),(nose_x_Coord+nose_width,nose_y_Coord+nose_height),(255,0,0),2)

            # Scale the mustache dimensions to detected nose 
            mustacheWidth =  3 * nose_width
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
 
            # Center the mustache on the bottom of the nose
            x_Coord1 = nose_x_Coord - (mustacheWidth/4)
            x_Coord2 = nose_x_Coord + nose_width + (mustacheWidth/4)
            y_Coord1 = nose_y_Coord + nose_height - (mustacheHeight/2)
            y_Coord2 = nose_y_Coord + nose_height + (mustacheHeight/2)
 
            # Check for clipping
            if x_Coord1 < 0:
                x_Coord1 = 0
            if y_Coord1 < 0:
                y_Coord1 = 0
            if x_Coord2 > width:
                x_Coord2 = width
            if y_Coord2 > height:
                y_Coord2 = height
 
            # Re-calculate the dimensions of the mustache image
            mustacheWidth = x_Coord2 - x_Coord1
            mustacheHeight = y_Coord2 - y_Coord1
 
            # Re-size the original image and the masks to the mustache dimensions calcualted above
            mustache = cv2.resize(mustacheImage, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            invertedMask = cv2.resize(invertedMask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
 
            # Take region of image for mustache from the background equal to the size of the mustache image
            roi = roiColor[y_Coord1:y_Coord2, x_Coord1:x_Coord2]
 
            # The original image only where the mustache is not in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(roi,roi,mask = invertedMask)
 
            # The image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
 
            # Combine the images
            dst = cv2.add(roi_bg,roi_fg)
 
            # Place the combined image back over the original image
            roiColor[y_Coord1:y_Coord2, x_Coord1:x_Coord2] = dst
            
            break

    out.write(frame)
    cv2.imshow('Video', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
webcam.release()
out.release()
cv2.destroyAllWindows()
