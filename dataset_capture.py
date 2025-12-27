import cv2
import os

# Create a directory to store the dataset if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    
# Initialize the camera [cite: 52]
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Load the built-in face detector (Haar Cascade) [cite: 30]
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input student details [cite: 73]
face_id = input('\n enter user id (e.g., 1, 2, 3) end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale for detection
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder [cite: 74]
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    # Stop after capturing 30 images
    k = cv2.waitKey(100) & 0xff 
    if k == 27: # Press 'ESC' to stop manually
        break
    elif count >= 30: 
         break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()