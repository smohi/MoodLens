import cv2
from deepface import DeepFace as df

#loading pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#initializing the webcam (0 is default cam)
cap = cv2.VideoCapture(0)

while True:
    #capture frame_by_frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break  # If frame not captured correctly, exit the loop
    
    #converting frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detecting face in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    #draw rectangle around face
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        
    #extracting face region from frame
    face_img = frame[y:y+h, x:x+w]
    
    try:
        #analyzing face using deepface to get attributes
        analysis = df.analyze(face_img, actions=['age', 'gender', 'emotion'], enforce_detection = False)
        
        #extracting age, gender, emotion 
        age = analysis['age']
        gender = analysis['gender']
        emotion = analysis['dominant_emotion']
        
        #preparing label for age, gender, emotion
        label = f"{gender}, {age}, {emotion}"
        label_position = (x, y + h + 20)
        
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),2)
    except Exception as e:
        print(f"Deepface analysis error: {e}")
        continue
        
    #displaying output
    cv2.imshow('Face Detection Analysis', frame)
    
    #break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#releasing capture and close window
cap.release()
cv2.destroyAllWindows
