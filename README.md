## THIS IS MY WORK WITH DROWSINESS DETECTION USING PYTHON ##
## FOLLOW THE INSTRUCTIONS BELOW TO GO ALONG WITH THE CODE ##

# face_recog_dlib_file dlib instalation error
 ------------------------------
 python 3.7 &
 ubuntu 16.04

 ------------------------------

 *1)* Assign root permission and update ubuntu first

 sudo su

 apt-get update


 *2)* Check your version & path for python & pip.

 which python3



 python3 -V



 which pip3



 pip3 -V


 *3)*

 pip3 install cmake


 *4)*

 apt-get install -y --fix-missing \

     build-essential \

     cmake \

     gfortran \

     git \

     wget \

     curl \

     grapgicsmagick \

     libgraphicsmagic-dev \

     libatlas-dev \

     libavcodec-dev \

     libavformat-dev \

     libgtk2.0-dev \

     libjpeg-dev \

     liblapack-dev \

     libswscale-dev \

     pkg-config \

     software-properties-common \

     zip


 *6)*

 apt-get install python3-dev


 *5)*

 pip3 install dlib


 ------------------------------

 ✅ Done


Note that after having python and pip updated you will have to download cmake using pip and just then, the dlib.



The project aims to trace similarities between people who are sleepy or about to sleep and identify whether they are still awake or not. Thus, if this were implemented in larger projects, it would be possible to alert passengers/vehicle drivers when the person driving was suspected of falling asleep.</br>

Project objective is:
Analyze and catalog images in 2 parts:</br>
-About to sleep/Asleep (Eyes constantly closed or with “sleepy” eyes)</br>
-Awake (Eyes open constantly)

First, it was necessary to trace the entire face using a library widely used for this type of need - Dlib.
With it we were able to draw an overview of the points on the face, covering 68 points in total.
<img src="https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg"/>


</br>
</br>

The dataset used for deep learning already comes from the library used and there are around 4700 images already converted into .DAT format. With this dataset and our algorithm it is possible to first trace the faces as below and then analyze whether the person is active or sleepy - which will be dealt with in the next slides.

</br>
How we are going to do this?</br>
What are the most common points if the person is about to sleep?
Not everyone yawns or has the head falls before falling asleep. Both vary greatly from person to person and situation to situation.
As such, the eyes are the main tool to focus on detection and will be treated as the main physiological sensors. However, there is the problem if
the driver is wearing sunglasses - which are mostly completely dark - and it is not possible to analyze the eyes.

</br>
There are currently more invasive solutions that are based on sensors attached to the body. There are several types of sensors and they are generally the most accurate for
identify whether the person is about to sleep or not. But there is also a problem with this, what measures should the person not use the devices? For this reason we focus our
algorithm for identifying drowsiness from the eyes, even though if a person is wearing glasses it is more difficult to identify.

</br>

Our first objective is to try to trace all 68 points on the face of the image person. And the good part is that the dlib library helps us a lot in this process. Following our code in the 'face.py' file you will see how easy it is. Anyway, with the predefined image you will have something like this:
</br>
<img src="https://rapidapi.com/blog/wp-content/uploads/2017/11/t01a5ed8aab97b460c9.jpg" />

</br>
</br>
<p>Then we can select just the points of the eyes and make our algorithm facing them. But what to do now? For this, we follow the  <a href="http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf"> academic document</a>
and we made the ratio of the top points to the bottom points of the eye. This reason pre-established by the students can make our analysis quite accurate.
</p>
That's basically what we did in the project. You can follow the "eyes.py" file to better understand how we coded.
</br>
<p>
To run the 4 types of images, all you need to do is change the references of the images to the other references that are imported in the initial lines of the project.
</br>
To run the test projects and accuracy check, you only need to run the specific file.
</p>
