import face_recognition
import cv2

# Load the image with the face you want to recognize
known_image = face_recognition.load_image_file("known_face.jpg")

# Encode the known face
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Capture video from the default camera (you can also provide a file path)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the known face
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the known face
        if matches[0]:
            name = "Jayson"

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
