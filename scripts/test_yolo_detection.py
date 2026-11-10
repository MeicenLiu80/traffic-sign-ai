# This script tests the saved YOLO detector on one original validation image.
# It uses a small inference size to reduce the computational load.
# It prints the detected bounding boxes and confidence scores in the terminal.
# It saves an annotated image only if the inference completes successfully.
# It does not load the CNN model or train either model.

import os
import random
import torch
from ultralytics import YOLO

# Step 1: Define the project folder and saved YOLO checkpoint.
# Step 1: Use a configurable local MTSD project folder instead of a personal absolute path.
project_folder = os.environ.get(
    "MTSD_PROJECT_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "local_mtsd"
    )
)

yolo_model_path = os.path.join(
    project_folder,
    "YOLO_Detection_Results",
    "yolo11n_traffic_sign_retry",
    "weights",
    "best.pt"
)

# Step 2: Define the original YOLO validation image folder.
validation_image_folder = os.path.join(
    project_folder,
    "YOLO_Detection_Dataset",
    "images",
    "val"
)

# Step 3: Check the required paths before loading the model.
if not os.path.isfile(yolo_model_path):
    raise FileNotFoundError(
        "The YOLO checkpoint was not found: " + yolo_model_path
    )

if not os.path.isdir(validation_image_folder):
    raise FileNotFoundError(
        "The validation image folder was not found: "
        + validation_image_folder
    )

print("YOLO checkpoint found.")
print("Validation image folder found.")

# Step 4: Find available image filenames without loading their contents.
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
validation_image_paths = []

for file_name in os.listdir(validation_image_folder):
    if file_name.lower().endswith(image_extensions):
        full_image_path = os.path.join(
            validation_image_folder,
            file_name
        )
        validation_image_paths.append(full_image_path)

if len(validation_image_paths) == 0:
    raise FileNotFoundError("No validation images were found.")

# Step 5: Select one image for this test.
random.seed(10)

selected_image_path = random.choice(validation_image_paths)

print("\nSelected image:")
print(selected_image_path)

# Step 6: Select the GPU when CUDA is available.
if torch.cuda.is_available():
    selected_device = 0
    print("\nUsing GPU:", torch.cuda.get_device_name(0))
else:
    selected_device = "cpu"
    print("\nCUDA is unavailable. Using CPU.")

# Step 7: Load the saved YOLO model.
print("\nLoading YOLO model...")

yolo_model = YOLO(yolo_model_path)

print("YOLO model loaded successfully.")

# Step 8: Run inference on one image with a small input size.
print("\nStarting YOLO inference...")

with torch.inference_mode():
    detection_results = yolo_model.predict(
        source=selected_image_path,
        imgsz=320,
        batch=1,
        conf=0.25,
        max_det=20,
        device=selected_device,
        verbose=False
    )

print("YOLO inference completed.")

# Step 9: Read the detection results.
first_result = detection_results[0]

if first_result.boxes is not None:
    number_of_detections = len(first_result.boxes)
else:
    number_of_detections = 0

print("\n========== YOLO DETECTION RESULTS ==========")
print("Number of detections:", number_of_detections)

if number_of_detections > 0:

    for detection_number, detected_box in enumerate(
        first_result.boxes,
        start=1
    ):
        box_coordinates = detected_box.xyxy[0].cpu().tolist()
        confidence_score = float(detected_box.conf[0].item())

        print("\nDetection number:", detection_number)
        print(
            "Bounding box:",
            [round(value, 2) for value in box_coordinates]
        )
        print(
            "Detection confidence:",
            round(confidence_score * 100, 2),
            "%"
        )

else:
    print("No traffic signs were detected in this image.")

# Step 10: Save the annotated image after successful inference.
output_folder = os.path.join(
    project_folder,
    "YOLO_Detection_Results",
    "terminal_test"
)

os.makedirs(output_folder, exist_ok=True)

output_image_path = os.path.join(
    output_folder,
    "yolo_detection_test.jpg"
)

annotated_image = first_result.plot()

import cv2

image_saved = cv2.imwrite(
    output_image_path,
    annotated_image
)

if image_saved:
    print("\nAnnotated image saved to:")
    print(output_image_path)
else:
    print("\nWarning: The annotated image could not be saved.")

print("\nYOLO standalone test finished.")

