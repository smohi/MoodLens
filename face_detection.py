import cv2

#loading pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#initializing the webcam (0 is default cam)
cap = cv2.VideoCapture(0)

while True:
    #capture frame_by_frame
    ret, frame = cap.read()
    
    #converting frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detecting face in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    #draw rectangle around face
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        
    #displaying output
    cv2.imshow('Face Detection', frame)
    
    #break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#releasing capture and close window
cap.release()
cv2.destroyAllWindows
