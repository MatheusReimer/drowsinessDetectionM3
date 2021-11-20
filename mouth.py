import cv2
import dlib
from scipy.spatial import distance
from imagesClosedEyes.figuraCollection import imageClosedEyes
from imagesOpenEyes.imagemCollection import imageOpenEyes
from imagesWithGlasses.imageWithGlassesCollection import withGlasses
from imagesWithSmile.withSmileCollection import withSmile



#GRABING THE MOUNTH OPOSITE POINTS AND SUBTRACTING THEM:
def calculateOpenMouth(side):
    A = distance.euclidean(side[1],side[7])
    B = distance.euclidean(side[2],side[6])
    C = distance.euclidean(side[3],side[5])
    D = distance.euclidean(side[0],side[4])
    ratio = (A+B+C)/(D*3)
    return ratio
 


##EXECUTING TWO DLIB FUNCTIONS TO READ THE IMAGE
##GETTING THE POINTS FROM THE DATASET
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


running = True
j=0

while running:
	##READING CURRENT IMAGE
    cap = cv2.imread(imageOpenEyes[j])
	  
    ##CONVERTING INTO GRAY IMAGE FOR BETTER PROCESSING
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
		   
    	###GETTING THE LANDMARKS POINTS FOR THE CURRENT FACE
        face_landmarks = dlib_facelandmark(gray, face)
        outside = []
        inside = []
		##EXECUTING FOR THE LEFT EYE
        for n in range(48,61):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	outside.append((x,y))
        	next_point = n+1
        	if n == 60:
        		next_point = 48
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(cap,(x,y),(x2,y2),(0,255,0),3)
		
        for n in range(60,68):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	inside.append((x,y))
        	next_point = n+1
        	if n == 67:
        		next_point = 60
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(cap,(x,y),(x2,y2),(0,255,0),3)

        ratio = calculateOpenMouth(inside)
        ratio = round(ratio,2)
        print(ratio)
        if ratio<0.1:
			##COVERING TEXT INTO THE IMAGE
        	cv2.putText(cap,"Boca Fechada!",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)

        	print("Sonolento")
        print(ratio)
        if ratio>=0.1:
            cv2.putText(cap,"Boca aberta/Sorrindo!",(20,100),
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