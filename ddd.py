import numpy as np
#Importing OpenCV Library for basic image processing functions
import cv2
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
from scipy.spatial import distance

#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#status marking for current state
sleep = 0; drowsy = 0; active = 0
status="Initial"
color=(0,0,0)


def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	eye_aspect_ratio = (A+B)/(2.0*C)
	return eye_aspect_ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    #detect face in faces array
    for face in faces:
        
        cv2.rectangle(frame, 
					(face.left(), face.top()), (face.right(), face.bottom()), 
					(0, 255, 0), 2)

        face_landmarks = dlib_facelandmark(gray, face)
        face_landmarks = face_utils.shape_to_np(face_landmarks)
		
        #The numbers are the eye landmarks
        left_ear = calculate_EAR(face_landmarks[36:42])
        right_ear = calculate_EAR(face_landmarks[42:48])

        ear = (left_ear+right_ear)/2
        ear = round(ear,2)
        
        # #Now judge what to do for the eye blinks
        if(ear<0.20):
        	sleep+=1
        	drowsy=0
        	active=0
        	if(sleep>6):
        		status="Sleeping !!!"
        		color = (255,0,0)

        elif(ear<0.25):
        	sleep=0
        	active=0
        	drowsy+=1
        	if(drowsy>6):
        		status="Drowsy !"
        		color = (0,0,255)

        else:
        	drowsy=0
        	sleep=0
        	active+=1
        	if(active>6):
        		status="Active !"
        		color = (0,255,0)

        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, color,3)

        for n in range(0, 68):
        	(x,y) = face_landmarks[n]
        	cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
   
    key = cv2.waitKey(1)
    if key == 32:
      	break
cap.release()
cv2.destroyAllWindows()

