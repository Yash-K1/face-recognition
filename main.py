import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

vansh_image = face_recognition.load_image_file("D:/Programs/face/photos/vansh.jpeg")
vansh_encoding = face_recognition.face_encodings(vansh_image)[0]

ayush_image = face_recognition.load_image_file("D:/Programs/face/photos/ayush.jpeg")
ayush_encoding = face_recognition.face_encodings(ayush_image)[0]

aman_image = face_recognition.load_image_file("D:/Programs/face/photos/aman.jpeg")
aman_encoding = face_recognition.face_encodings(aman_image)[0]

prab_image = face_recognition.load_image_file("D:/Programs/face/photos/prab.jpeg")
prab_encoding = face_recognition.face_encodings(prab_image)[0]

known_face_encoding = [
    vansh_encoding,
    ayush_encoding,
    aman_encoding,
    prab_encoding
]

known_faces_names = [
    "vansh",
    "ayush",
    "aman",
    "prabhakar"
]


students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file outside the loop
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Ensure RGB format (channels first)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if s:
        # Try adjusting tolerance (looser detection)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(face_locations)  # Print for debugging

        if face_locations:
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_landmarks_list)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]

                face_names.append(name)

                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 100)
                    fontScale = 1.5
                    fontColor = (255, 0, 0)
                    thickness = 3
                    lineType = 2

                    cv2.putText(frame, name + ' Present',
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)

                    if name in students:
                        students.remove(name)
                        print(students)
                        current_time = now.strftime("%H-%M-%S")
                        lnwriter.writerow([name, current_time])

    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()


