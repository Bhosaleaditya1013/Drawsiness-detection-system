# Drawsiness Detection system : 

# Drowsiness refers to "feeling more sleepy than normal during the day"...

'''Application Code : '''


# I imports some libraries for working system: 
from scipy.spatial import distance  # It Calculates the distance between two points in space, used for eye aspect ratio (EAR) calculation.
from imutils import face_utils      # It Provides easy image processing functions, like resizing.
from pygame import *                # It is Used to load and play sounds (alerts).
import imutils                      
import dlib                         # It Handles face detection and facial landmark prediction.
import cv2                          # It Deals with real-time video capture and processing.


# by using pygame library
mixer.init()
mixer.music.load("music.wav")
'''This sets up the pygame.mixer to play the alert sound. The alert sound file "music.wav" is loaded so that it can be played when the person is sleepy.'''

# making one function by which i can return eye size or aspect ratio
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	EAR = (A + B) / (2.0 * C)        #EAR= Eye Aspect Ratio
	return EAR
'''EAR measures how "open" or "closed" the eyes are.
   The function calculates the vertical distances between key eye landmarks and compares them to the horizontal distance.'''


# If EAR is below a certain threshold for a certain amount of frames, the person is likely Sleepy:
threshold = 0.25                # The threshold below which the eye aspect ratio indicates closed eyes.
frame_check = 20                # The number of consecutive frames with closed eyes needed before an alert is triggered.



detect = dlib.get_frontal_face_detector()        # It is a face detector using dlib, which detects faces in the video.
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  
'''This loads a pre-trained model (shape_predictor_68_face_landmarks.dat) that can predict the position of 68 landmarks on the face, including eye landmarks.'''
'''this can download from Dlib's official site '''

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]  # This gives you the start and end indices for the left eye landmarks.
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] # This gives you the start and end indices for the right eye landmarks.

 
capture=cv2.VideoCapture(0) # The video is captured from the camera (cap = cv2.VideoCapture(0)), and each frame is processed in a loop. Laptop frontcamera is cv2.VideoCapture(0) ...if there is external camera then cv2.VideoCapture(1)


flag=0   # creating one flag

# Main Logic :
while True:
	ret, frame=capture.read()
	frame = imutils.resize(frame, width=750)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < threshold:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
    
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
capture.release()