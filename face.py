from typing_extensions import runtime
import cv2
import dlib

folder= "./imagesOpenEyes/"

imageOpenEyes =[
folder+"imagem1.jpg",
folder+"imagem2.jpg",
folder+"imagem3.jpg",
folder+"imagem4.jpg",
folder+"imagem5.jpg",
folder+"imagem6.jpg",
folder+"imagem7.jpg",
folder+"imagem8.jpg",
folder+"imagem9.jpg",
folder+"imagem10.jpg",
folder+"imagem11.jpg",
folder+"imagem12.jpg",
folder+"imagem13.jpg",
folder+"imagem14.jpg",
folder+"imagem15.jpg"]


hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
running = True
j=0

while running:
    cap = cv2.imread(imageOpenEyes[j])
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(cap, (x, y), 1, (0, 69, 255), 10)



               

    key = cv2.waitKey(1)
 
    if key == 27:
        j = j+1

cv2.destroyAllWindows()