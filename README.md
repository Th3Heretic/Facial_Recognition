The following modules will need to be installed in the terminal in order for the "FacialRecognition" programs to work:

>pip install face_recognition numpy opencv-python dlib imutils tensorflow emotion_recognition

>pip install --upgrade emotion_recognition numpy tensorflow

Or for python 3 users, try:
>python3 -m pip ...

Functionality has been updated to allow the program to use the default system camera to read your facial data. 

If it does not have a record of your face, it will take a screenshot of the frame where it identified your face, label it 
as 'unknown' with the date and time the picture was captured, your name will then appear as 'unknown' above your face until you
rename the screenshot file as [name].jpg

A corresponding emotion will also be displayed next to your name separated with a hyphen.

The Python file named 'EmotionDetection.py' is just a basic program to detect and display the determined facial 
expression without the name identifier and as such, has no saved data-model file implemented into it.

[Full commit history unavailable as the GitHub repository was only created as an afterthought to share the code with friends]

Known Issues:
Returns a list index error when encountered with a new face following a freeze and crash.
    Fix: Returns 'Unknown' above the target instead of asking for their name.
