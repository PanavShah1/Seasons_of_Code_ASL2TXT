import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import matplotlib.pyplot as plt

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)
BUFFER_SPACE = 35
IMAGE_DIR = "images/woman_hands.jpg"

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    annotated_image = np.copy(rgb_image)   
    x_min, x_max, y_min, y_max = 0, rgb_image.shape[1], 0, rgb_image.shape[0]
    handedness = None
    all_coordinates = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        try:
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Collect the coordinates of each landmark
            coordinates = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks]
            all_coordinates.append(coordinates)

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style()
            )
        
            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            x_min = max(int(min(x_coordinates) * width) - BUFFER_SPACE, 0)
            x_max = min(int(max(x_coordinates) * width) + BUFFER_SPACE, width)
            y_min = max(int(min(y_coordinates) * height) - BUFFER_SPACE, 0)
            y_max = min(int(max(y_coordinates) * height) + BUFFER_SPACE, height)

            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Draw a blue bounding box

            text_x = x_min
            text_y = y_min - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        except Exception as e:
            print(f"Error processing hand landmarks: {e}")

    return {
        "annotated_image": annotated_image,
        "coordinates": [x_min, x_max, y_min, y_max],
        "handedness": handedness,
        "all_coordinates": all_coordinates
    }


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def get_hand_landmarker(image_dir):
    image = mp.Image.create_from_file(image_dir)
    image_RGB = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    
    detection_result = detector.detect(image)
    annoted_image = draw_landmarks_on_image(image_RGB, detection_result)["annotated_image"]
    return annoted_image

def show(image_dir):
    image = mp.Image.create_from_file(image_dir)
    image_RGB = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
    
    detection_result = detector.detect(image)
    annoted_image = draw_landmarks_on_image(image_RGB, detection_result)["annotated_image"]
    plt.subplot(1, 2, 1)
    plt.imshow(image_RGB)
    plt.subplot(1, 2, 2)
    plt.imshow(annoted_image)
    plt.show()

if __name__ == "__main__":
    show(IMAGE_DIR)