import cv2
import dlib
from scipy.spatial import distance
from imagesClosedEyes.figuraCollection import imageClosedEyes
from imagesOpenEyes.imagemCollection import imageOpenEyes
from imagesWithGlasses.imageWithGlassesCollection import withGlasses
from imagesWithSmile.withSmileCollection import withSmile
import os

# EAR BEING CALCULATED BY THE DOCS PROVIDED HERE:
# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


# GRABING THE MOUNTH OPOSITE POINTS AND SUBTRACTING THEM:
def calculateOpenMouth(side):
    A = distance.euclidean(side[1], side[7])
    B = distance.euclidean(side[2], side[6])
    C = distance.euclidean(side[3], side[5])
    D = distance.euclidean(side[0], side[4])
    ratio = (A+B+C)/(D*3)
    return ratio


# EXECUTING TWO DLIB FUNCTIONS TO READ THE IMAGE
# GETTING THE POINTS FROM THE DATASET
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")

# SETTING CONDITIONAL VARIABLES
running = True
areTheEyesClosed = False
imageCounter = 0
accuracyCounter = 0
while running:
    # READING CURRENT IMAGE
    
    path = "M:/CienciaDaComputacao/ProcessamentoDeImagem/M2/Tests/FaceImages/FatigueSubjects/"
    for image in os.listdir(path):
      
        cap = cv2.imread(path+image)

        # CONVERTING INTO GRAY IMAGE FOR BETTER PROCESSING
        gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        for face in faces:

            # GETTING THE LANDMARKS POINTS FOR THE CURRENT FACE
            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []
            ###################################################################################
            # EYES
            ###################################################################################
            # EXECUTING FOR THE LEFT EYE
            for n in range(36, 42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x, y))
                next_point = n+1
                # THIS CONDITION IS HERE TO LIMIT THE DRAW TO ONE EYE
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(cap, (x, y), (x2, y2), (0, 255, 0), 3)
                # EXECUTING FOR THE RIGHT EYE
            for n in range(42, 48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x, y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(cap, (x, y), (x2, y2), (0, 255, 0), 3)
                # GETTING EAR FOR BOTH EYES AND THAN MAKING AN AVERAGE
            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)
            EAR = (left_ear+right_ear)/2
            EAR = round(EAR, 2)
            toPrint = f"EAR = {EAR}  Are the Eyes Closed? False"
            
            if EAR < 0.26:

                
                areTheEyesClosed = True
            ###################################################################################
                # MOUNTH
            ###################################################################################
                if areTheEyesClosed == True:
                    outside = []
                    inside = []

                    # GRABING THE OUTSIDE POINTS OF THE MOUNTH
                    for n in range(48, 61):
                        x = face_landmarks.part(n).x
                        y = face_landmarks.part(n).y
                        outside.append((x, y))
                        next_point = n+1
                        if n == 60:
                            next_point = 48
                        x2 = face_landmarks.part(next_point).x
                        y2 = face_landmarks.part(next_point).y
                        cv2.line(cap, (x, y), (x2, y2), (0, 255, 0), 3)
                        # GRABING THE INSIDE POINT OF THE MOUNTH
                    for n in range(60, 68):
                        x = face_landmarks.part(n).x
                        y = face_landmarks.part(n).y
                        inside.append((x, y))
                        next_point = n+1
                        if n == 67:
                            next_point = 60
                        x2 = face_landmarks.part(next_point).x
                        y2 = face_landmarks.part(next_point).y
                        cv2.line(cap, (x, y), (x2, y2), (0, 255, 0), 3)

                    ratio = calculateOpenMouth(inside)
                    ratio = round(ratio, 2)
                    toPrintMounthRatio = f"MOUNTH RATIO = {ratio}"

                
                    if ratio >= 0.1:

                        
                        areTheEyesClosed = False
                        

                    



        
        if areTheEyesClosed == True:
            accuracyCounter = accuracyCounter +1
            counterToPrint = f"{accuracyCounter} / {imageCounter} imagem: {image}"
            print(counterToPrint)
        imageCounter=imageCounter+1
        if imageCounter == 4560:
            break
      

cv2.destroyAllWindows()
