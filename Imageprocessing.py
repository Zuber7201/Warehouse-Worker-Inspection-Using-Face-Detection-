import cv2
import dlib
import numpy as np
import pickle
import pyttsx3  # For text-to-speech conversion
import os
from datetime import datetime

# Load Dlib's face detector and models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("/Users/icentral/Desktop/miniproject/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("/Users/icentral/Desktop/miniproject/dlib_face_recognition_resnet_model_v1.dat")

# Load worker embeddings
embeddings_file = "worker_embeddings.pkl"
if not os.path.exists(embeddings_file):
    print(f"Error: Embeddings file '{embeddings_file}' not found. Make sure you have trained workers' embeddings.")
    exit()

with open(embeddings_file, "rb") as f:
    worker_embeddings = pickle.load(f)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Adjust speaking speed
engine.setProperty("volume", 1.0)  # Adjust volume (1.0 is max)

# Function to recognize a worker in a frame
def recognize_worker(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame, 1)
    recognized_workers = []

    for face in faces:
        shape = shape_predictor(rgb_frame, face)
        descriptor = face_rec_model.compute_face_descriptor(rgb_frame, shape)
        descriptor = np.array(descriptor)

        # Compare with known embeddings
        min_distance = float("inf")
        best_match = None
        for worker_name, worker_embedding in worker_embeddings.items():
            distance = np.linalg.norm(descriptor - worker_embedding)
            if distance < 0.6 and distance < min_distance:  # Threshold of 0.6
                min_distance = distance
                best_match = worker_name

        # Draw face bounding box and label if recognized
        if best_match:
            recognized_workers.append(best_match)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, best_match, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Draw bounding box for unrecognized faces
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, recognized_workers

# Function to ask questions to recognized workers and save responses
def ask_questions(worker_name):
    questions = [
        "How are you feeling today? Any health issues?",
        "Have you followed the safety protocols today?",
        "Do you have any advice for improving warehouse operations?"
    ]

    # Create a folder for the worker with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    worker_folder = f"./worker_responses/{worker_name}_{timestamp}"
    os.makedirs(worker_folder, exist_ok=True)

    # File to save the responses
    response_file = os.path.join(worker_folder, "responses.txt")

    # Read out each question using text-to-speech and save answers
    print(f"\nQuestions for {worker_name}:")
    with open(response_file, "w") as f:
        for question in questions:
            engine.say(f"Hello {worker_name}, {question}")
            engine.runAndWait()
            print(question)
            answer = input("Your answer: ")
            print(f"Recorded: {answer}")
            f.write(f"Question: {question}\nAnswer: {answer}\n\n")

    print(f"Responses saved in: {response_file}")

# Real-time recognition with webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

already_asked = set()  # Keep track of workers who have been asked questions during the session

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Recognize workers in the current frame
    frame, recognized_workers = recognize_worker(frame)

    # If workers are recognized, ask them questions (only once per session)
    for worker in recognized_workers:
        if worker not in already_asked:
            ask_questions(worker)
            already_asked.add(worker)

    # If no workers are recognized
    if not recognized_workers:
        engine.say("Person not in our data.")
        engine.runAndWait()
        print("Person not in our data.")
        cv2.putText(frame, "Person not in our data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with recognition results
    cv2.imshow("Worker Recognition", frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()