import cv2

image_path = "IMAGE_PATH"
cascade_path = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

#Delete or comment lines 9-23 if you want to run the script only for the webcam (without a tester image)
image = cv2.imread(image_path) 

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_faces = face_cascade.detectMultiScale(
    image_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

for(x,y,w,h) in image_faces: 
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,2), (2))

print("Found {0} faces!".format(len(image_faces)))
cv2.imshow('Image', image)

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    video_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    video_faces = face_cascade.detectMultiScale(
        video_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for(x,y,w,h) in video_faces: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), (2))
   
    print("Found {0} faces!".format(len(video_faces)))

    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
video_capture.release()
cv2.destroyAllWindows()