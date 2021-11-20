import cv2
import dlib
from scipy.spatial import distance
from imagesClosedEyes.figuraCollection import imageClosedEyes
from imagesOpenEyes.imagemCollection import imageOpenEyes
from imagesWithGlasses.imageWithGlassesCollection import withGlasses
from imagesWithSmile.withSmileCollection import withSmile



# EAR BEING CALCULATED BY THE DOCS PROVIDED HERE:
#http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

##EXECUTING TWO DLIB FUNCTIONS TO READ THE IMAGE
##GETTING THE POINTS FROM THE DATASET
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


running = True
j=0

while running:
	##READING CURRENT IMAGE
    cap = cv2.imread(withSmile[j])
	  
    ##CONVERTING INTO GRAY IMAGE FOR BETTER PROCESSING
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
		   
    	###GETTING THE LANDMARKS POINTS FOR THE CURRENT FACE
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
		##EXECUTING FOR THE LEFT EYE
        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
			## THIS CONDITION IS HERE TO LIMIT THE DRAW TO ONE EYE
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(cap,(x,y),(x2,y2),(0,255,0),3)
		##EXECUTING FOR THE RIGHT EYE
        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(cap,(x,y),(x2,y2),(0,255,0),3)
		##GETTING EAR FOR BOTH EYES AND THAN MAKING AN AVERAGE
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
		##USING 0.26 BECAUSE OF THE CECH AND SOUKUPOVÁ STUDIES
        if EAR<0.26:
			##COVERING TEXT INTO THE IMAGE
        	cv2.putText(cap,"SONOLENTO!",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        	cv2.putText(cap,"VOCE ESTA DORMINDO?",(20,400),
        		cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        	print("Sonolento")
        print(EAR)
        if EAR>=0.26:
            cv2.putText(cap,"ATIVO!",(20,100),
        	    cv2.FONT_HERSHEY_SIMPLEX,3,(0,255,0),4)
            print("Ativo")
	##RESIZING IMAGE FOR BETTER VIEW
    imS = cv2.resize(cap, (500, 900))  
	##PRINTING IMAGE   
    cv2.imshow("Classification", imS)    

    key = cv2.waitKey(1)
	##27 = ESC KEY
    if key == 27:
        j = j+1

cv2.destroyAllWindows()