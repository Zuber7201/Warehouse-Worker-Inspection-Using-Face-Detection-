import dlib
import cv2
import os
import numpy as np
import pickle

# Load Dlib's modelsdat
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("/Users/icentral/Desktop/miniproject/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("/Users/icentral/Desktop/miniproject/dlib_face_recognition_resnet_model_v1.dat")

# Function to encode a face
def encode_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image, 1)
    if len(faces) == 0:
        return None
    shape = shape_predictor(rgb_image, faces[0])  # Assume one face per image
    face_descriptor = face_rec_model.compute_face_descriptor(rgb_image, shape)
    return np.array(face_descriptor)

# Load and encode workers
def encode_workers(dataset_path):
    embeddings = {}
    for worker in os.listdir(dataset_path):
        worker_path = os.path.join(dataset_path, worker)
        if os.path.isdir(worker_path):
            for image_name in os.listdir(worker_path):
                image_path = os.path.join(worker_path, image_name)  # Full path to each image
                embedding = encode_face(image_path)
                if embedding is not None:
                    embeddings[worker] = embedding
                    break  # One image per worker is sufficient
    return embeddings

if __name__ == "__main__":
    dataset_path = "/Users/icentral/Desktop/miniproject/dataset"  # Your dataset path
    embeddings = encode_workers(dataset_path)
    with open("worker_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    print("Worker embeddings saved!")