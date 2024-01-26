The following modules will need to be installed in the terminal in order for the "FacialRecognition" programs to work:

>pip install face_recognition numpy opencv-python dlib imutils tensorflow emotion_recognition

>pip install --upgrade emotion_recognition numpy tensorflow

Functionality has been updated to allow the program to use the default system camera to read your facial data. 

If it does not have a record of your face, it will ask for your name then screenshot and save that screenshot under 
that name to be called later when identifying you.

A corresponding emotion will also be displayed next to your name separated with a hyphen.

The Python file named 'EmotionDetection.py' is just a basic program to detect and display the determined facial 
expression without the name identifier and as such, has no saved model file implemented into it.

[Full commit history unavailable as the GitHub repository was only created as an afterthought to share the code with friends]