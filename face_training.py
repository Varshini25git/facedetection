import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Get the filename (e.g., "User.1.1.jpg")
        filename = os.path.split(imagePath)[-1]

        # 1. Skip system files or hidden files
        if filename.startswith("."):
            continue

        # 2. Check if the filename follows the format "User.ID.Count.jpg"
        parts = filename.split(".")
        if len(parts) < 3 or parts[0] != "User":
            print(f"[WARNING] Skipping file '{filename}' - incorrect format.")
            continue

        try:
            # 3. Load image and convert to grayscale
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')

            # 4. Extract the user ID safely
            id = int(parts[1])
            
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        except Exception as e:
            print(f"[ERROR] Could not process {filename}: {e}")

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

# Run the training
faces, ids = getImagesAndLabels(path)

# Check if any faces were actually found
if len(faces) == 0:
    print("\n [ERROR] No training data found! Please run Step 1 (Capture) first.")
else:
    recognizer.train(faces, np.array(ids))

    # Save the model
    if not os.path.exists('trainer'):
        os.makedirs('trainer')
    recognizer.write('trainer/trainer.yml') 

    print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")