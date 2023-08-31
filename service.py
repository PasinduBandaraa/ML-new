import cv2
import os
import face_recognition
import pickle
import joblib

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        return
    # Get the frames per second (fps) and frame width/height
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Initialize frame counter
    frame_count = 0
    count = 0
    # Loop through each frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save the frame as an image file
        try:
            frame_count += 1
            face_recognition.face_encodings(frame)[0]  # Assuming one face per image
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            count +=1
            if count>5:
                break
        except IndexError:
            print("no faces found for frame"+str(frame_count))
    # Release the video capture object
    cap.release()
    collect_known_faces()


def collect_known_faces():
    known_encodings = []
    known_names = []
    dataset_path = "faces"
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            try:
                face_encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
                known_encodings.append(face_encoding)
                known_names.append(person_name)
            except IndexError:
                print(image_path)

    with open("known_encodings.pkl", "wb") as f_encodings:
        pickle.dump(known_encodings, f_encodings)

    with open("known_names.pkl", "wb") as f_names:
        pickle.dump(known_names, f_names)


def load_known_faces():
    with open("known_encodings.pkl", "rb") as f_encodings:
        known_encodings = pickle.load(f_encodings)
    with open("known_names.pkl", "rb") as f_names:
        known_names = pickle.load(f_names)
    return known_encodings, known_names

def identify_faces(image_path, known_encodings, known_names):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]

    return name

loaded_model = joblib.load("severity_classifier.pkl")

def get_severity(data):
    return loaded_model.predict([data])[0]