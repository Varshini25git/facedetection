import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time

# --- CONFIGURATION ---
# Your Names List (Ensure order matches IDs: 1=Uday, 2=Siva, etc.)
names = ['Uday', 'Siva', 'Varshini'] 
RUNTIME_SECONDS = 30  # System will run for this many seconds
# ---------------------

# 1. Setup Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
if not os.path.exists('trainer/trainer.yml'):
    print("[ERROR] 'trainer/trainer.yml' not found. Please run Step 2 first.")
    exit()

recognizer.read('trainer/trainer.yml')
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

# 2. Setup Camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

marked_students = set()
filename = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Load existing attendance (optional, just to avoid re-marking)
if os.path.exists(filename):
    df = pd.read_csv(filename)
    existing = df['Name'].unique()
    for name in existing:
        marked_students.add(name)

def mark_attendance(name):
    if name not in marked_students:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=['Name', 'Time'])
            df.to_csv(filename, index=False)
            
        new_entry = pd.DataFrame({'Name': [name], 'Time': [dtString]})
        new_entry.to_csv(filename, mode='a', header=False, index=False)
        
        marked_students.add(name)
        print(f" [SUCCESS] Attendance Marked for {name}")

print(f"\n [INFO] System starting. Will close automatically in {RUNTIME_SECONDS} seconds...")
start_time = time.time()

while True:
    # --- TIME CHECK ---
    elapsed_time = time.time() - start_time
    remaining_time = int(RUNTIME_SECONDS - elapsed_time)
    
    if elapsed_time > RUNTIME_SECONDS:
        print("\n [INFO] Time limit reached! Closing system.")
        break

    ret, img = cam.read()
    if not ret: break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 100):
            if id < len(names):
                name = names[id]
            else:
                name = "Unknown"
            
            confidence_text = f"  {round(100 - confidence)}%"
            
            # Mark attendance if confidence is good
            if (100 - confidence) > 40:
                mark_attendance(name)
        else:
            name = "unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    # Display Timer on Screen
    cv2.putText(img, f"Closing in: {remaining_time}s", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27: # Press 'ESC' to exit early
        break

cam.release()
cv2.destroyAllWindows()