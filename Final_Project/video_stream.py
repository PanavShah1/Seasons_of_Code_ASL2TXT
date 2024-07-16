import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from hand_landmarker import draw_landmarks_on_image
from rembg import remove
import torch
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
from utils import num_to_label, label_to_num
from Model import TinyVGG

# Load the model once
model = TinyVGG(3, 36, 16)
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),  
])

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

while cv2.waitKey(1) != 27:  # Escape key
    has_frame, frame = source.read()
    if not has_frame:
        break

    image_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_RGB)
    detection_result = detector.detect(image)
    result = draw_landmarks_on_image(image_RGB, detection_result)

    if result["annotated_image"] is not None and len(result["coordinates"]) == 4:
        annotated_image = result["annotated_image"]
        [x_min, x_max, y_min, y_max] = result["coordinates"]
        image_crop = image_RGB[y_min:y_max, x_min:x_max]
        # image_bg_rem = remove(image_crop)
        image_bg_rem = image_crop
        image_bg_black = np.copy(image_bg_rem)
        # image_bg_black[np.all(image_bg_black == [0, 0, 0, 0], axis=-1)] = [0, 0, 0, 255]

        image_tensor = transform(Image.fromarray(image_bg_black, 'RGB')).unsqueeze(0)
        with torch.inference_mode():
            pred = model(image_tensor)
            prob = torch.softmax(pred, dim=1)
            label = torch.argmax(prob, dim=1)
        
        # Get the label text
        label_text = num_to_label[label.item()]
        try:
            cv2.putText(annotated_image, label_text, (x_min, y_min + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            confidence_score = float(torch.max(prob).item())
            cv2.putText(annotated_image, f"{confidence_score:.2f}", (x_min + 20, y_min + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error displaying text: {e}")
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_name, annotated_image)
    else:
        cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
