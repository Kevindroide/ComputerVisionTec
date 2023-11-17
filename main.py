import cv2
import numpy as np
import face_recognition


# Initialize the webcam
cap = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

n = 3  # Number of faces to capture

for i in range(n):
    while True:
        # Capture the frame
        ret, frame = cap.read()

        # Display the frame for preview
        cv2.imshow(f'Press "c" to capture face {i+1}', frame)

        # If 'c' is pressed, break the loop and capture the photo
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            break

    # Save the captured frame
    cv2.imwrite(f"captured_image_{i+1}.jpg", frame)

    # Load the captured image
    image = face_recognition.load_image_file(f"captured_image_{i+1}.jpg")

    # Extract facial features of the captured image
    image_encoding = face_recognition.face_encodings(image)[0]

    # Add the face encoding and name to the known faces lists
    known_face_encodings.append(image_encoding)
    known_face_names.append(f"Person {i+1}")
    

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

   # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the facial features of the detected face with the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match is found, label the face with the name of the person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)
        else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)



    # Display the output image
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()