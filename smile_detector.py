import cv2

# Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

# Webcam Feed
webcam = cv2.VideoCapture(0)

# Pull current frame
while True:

    #Read webcam
    successfull_frame_read, frame = webcam.read()

    # If there is an error break
    if not successfull_frame_read:
        break

    # Convert to Gray Scale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #
    faces = face_detector.detectMultiScale(frame_grayscale)
    

    # Run face Detection 
    for(x, y, w, h) in faces:

        # Draw a rectangle
        cv2.rectangle(frame, (x, y), (w + w, y + h), (100, 200, 50), 4)

        # Get the sub frame (N-Dimensional slicing)
        the_face = frame[y: y + h, x: x + w]
    
        # convert to gray scale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # checkl every scale possible for image
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        eye = eye_detector.detectMultiScale(face_grayscale)

        #Find all smiles
        for(x_, y_, w_, h_) in smiles:

            cv2.rectangle(the_face, (x_, y_), (x_ +w_, y_ + h_), (50, 50, 200), 4)

        #Find all eyes
        for(x_, y_, w_, h_) in eye:

            cv2.rectangle(the_face, (x_, y_), (x_ +w_, y_ + h_), (50, 50, 255), 4)

        # Put smiling on the video if smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y + h + 40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))




    # Show each farme of feed
    cv2.imshow('Smile Detector', frame)

    # Display
    cv2.waitKey(1)



# Terminate camera read
webcam.release()

# Close all wondows
cv2.destroyAllWindows()

# code will rin wothout errors
print("code complete")