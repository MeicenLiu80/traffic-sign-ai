# This script evaluates the integrated YOLO and CNN pipeline on 20 validation images.
# It records detected bounding boxes, predicted classes, and confidence scores.
# It saves annotated images and a CSV file containing the prediction records.
# It uses the existing model weights and does not train or modify either model.

import os
import json
import csv
import random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import cv2

# Step 1: Define the project folder and the saved model paths.
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

cnn_model_path = os.path.join(
    project_folder,
    "CNN_Data_Augmentation_Results",
    "cnn_augmentation_best.pth"
)

class_mapping_path = os.path.join(
    project_folder,
    "train",
    "final_class_to_index.json"
)

validation_image_folder = os.path.join(
    project_folder,
    "YOLO_Detection_Dataset",
    "images",
    "val"
)

# Step 2: Define the output folders and result file.
output_folder = os.path.join(
    project_folder,
    "YOLO_Detection_Results",
    "batch_test_20"
)

annotated_image_folder = os.path.join(
    output_folder,
    "annotated_images"
)

csv_output_path = os.path.join(
    output_folder,
    "batch_test_results.csv"
)

os.makedirs(annotated_image_folder, exist_ok=True)

# Step 3: Check the required files and folders.
required_paths = [
    yolo_model_path,
    cnn_model_path,
    class_mapping_path,
    validation_image_folder
]

for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError(
            "Required file or folder not found: " + required_path
        )

print("All required files and folders were found.")

# Step 4: Select the available computing device.
if torch.cuda.is_available():
    selected_device = 0
    torch_device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    selected_device = "cpu"
    torch_device = torch.device("cpu")
    print("CUDA is unavailable. Using CPU.")

# Step 5: Define the CNN architecture matching the saved weights.
class TrafficSignCNN(nn.Module):

    def __init__(self, number_of_classes=326):
        super(TrafficSignCNN, self).__init__()

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classification_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, input_images):
        extracted_features = self.convolutional_layers(input_images)
        output_scores = self.classification_layers(extracted_features)
        return output_scores

# Step 6: Load the saved class mapping.
with open(class_mapping_path, "r", encoding="utf-8") as mapping_file:
    class_to_index = json.load(mapping_file)

index_to_class = {
    int(class_index): class_name
    for class_name, class_index in class_to_index.items()
}

number_of_classes = len(class_to_index)

print("Number of CNN classes:", number_of_classes)

# Step 7: Load the YOLO detector.
print("\nLoading YOLO model...")
yolo_model = YOLO(yolo_model_path)
print("YOLO model loaded successfully.")

# Step 8: Load the CNN classifier.
print("\nLoading CNN model...")
cnn_model = TrafficSignCNN(number_of_classes=number_of_classes)

cnn_weights = torch.load(
    cnn_model_path,
    map_location="cpu",
    weights_only=True
)

cnn_model.load_state_dict(cnn_weights, strict=True)
cnn_model = cnn_model.to(torch_device)
cnn_model.eval()

print("CNN model loaded successfully.")

# Step 9: Define the CNN preprocessing used in our preliminary tests.
cnn_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Step 10: Collect and randomly select 20 validation images.
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
validation_image_paths = []

for file_name in os.listdir(validation_image_folder):
    if file_name.lower().endswith(image_extensions):
        validation_image_paths.append(
            os.path.join(validation_image_folder, file_name)
        )

if len(validation_image_paths) == 0:
    raise FileNotFoundError("No validation images were found.")

number_of_images_to_test = min(20, len(validation_image_paths))

random.seed(42)

selected_image_paths = random.sample(
    validation_image_paths,
    number_of_images_to_test
)

print("\nNumber of selected images:", number_of_images_to_test)

# Step 11: Prepare a list to store all prediction records.
all_prediction_records = []

total_detected_objects = 0
images_with_detections = 0

# Step 12: Process the selected images one by one.
for image_number, image_path in enumerate(
    selected_image_paths,
    start=1
):

    print("\nProcessing image", image_number, "of", number_of_images_to_test)
    print("Image name:", os.path.basename(image_path))

    # Read the original image for cropping and annotation.
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Warning: Image could not be read. Skipping this image.")
        continue

    image_height, image_width = original_image.shape[:2]

    # Convert the image to RGB for the CNN and drawing workflow.
    original_rgb_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    # Run YOLO on one image at a time to keep the workload controlled.
    with torch.inference_mode():
        detection_results = yolo_model.predict(
            source=image_path,
            imgsz=320,
            batch=1,
            conf=0.25,
            max_det=20,
            device=selected_device,
            verbose=False
        )

    first_result = detection_results[0]

    if first_result.boxes is not None:
        detected_boxes = first_result.boxes
    else:
        detected_boxes = []

    number_of_detections = len(detected_boxes)

    total_detected_objects += number_of_detections

    if number_of_detections > 0:
        images_with_detections += 1

    # Make a separate copy for drawing the results.
    annotated_image = original_image.copy()

    # Record an explicit row for images with no detected objects.
    if number_of_detections == 0:
        all_prediction_records.append({
            "image_name": os.path.basename(image_path),
            "detection_number": 0,
            "predicted_class": "No detection",
            "yolo_confidence": "",
            "cnn_confidence": "",
            "x_min": "",
            "y_min": "",
            "x_max": "",
            "y_max": ""
        })

    # Step 13: Crop and classify every detected object.
    for detection_number, detected_box in enumerate(
        detected_boxes,
        start=1
    ):

        box_coordinates = detected_box.xyxy[0].cpu().tolist()

        x_min = int(box_coordinates[0])
        y_min = int(box_coordinates[1])
        x_max = int(box_coordinates[2])
        y_max = int(box_coordinates[3])

        # Keep the bounding box inside the image boundaries.
        x_min = max(0, min(x_min, image_width - 1))
        y_min = max(0, min(y_min, image_height - 1))
        x_max = max(x_min + 1, min(x_max, image_width))
        y_max = max(y_min + 1, min(y_max, image_height))

        # Crop the detected object from the original RGB image.
        cropped_array = original_rgb_image[y_min:y_max, x_min:x_max]

        if cropped_array.size == 0:
            print("Warning: Empty crop encountered. Skipping this detection.")
            continue

        cropped_sign = Image.fromarray(cropped_array).convert("RGB")

        # Resize the crop and convert it to a tensor for the CNN.
        cnn_input = cnn_image_transform(cropped_sign).unsqueeze(0)
        cnn_input = cnn_input.to(torch_device)

        # Run the CNN classifier.
        with torch.inference_mode():
            classification_scores = cnn_model(cnn_input)

            classification_probabilities = torch.softmax(
                classification_scores,
                dim=1
            )

            class_confidence, predicted_class_index = torch.max(
                classification_probabilities,
                dim=1
            )

        predicted_index = int(predicted_class_index.item())
        predicted_class_name = index_to_class[predicted_index]

        cnn_confidence_value = float(class_confidence.item())
        yolo_confidence_value = float(detected_box.conf[0].item())

        # Save this detection in the results list.
        all_prediction_records.append({
            "image_name": os.path.basename(image_path),
            "detection_number": detection_number,
            "predicted_class": predicted_class_name,
            "yolo_confidence": round(yolo_confidence_value, 4),
            "cnn_confidence": round(cnn_confidence_value, 4),
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })

        # Draw the bounding box.
        cv2.rectangle(
            annotated_image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # Create the displayed prediction label.
        prediction_label = (
            predicted_class_name
            + " | YOLO "
            + str(round(yolo_confidence_value * 100, 1))
            + "%"
            + " | CNN "
            + str(round(cnn_confidence_value * 100, 1))
            + "%"
        )

        # Draw the label near the top of the bounding box.
        label_y_position = max(20, y_min - 8)

        cv2.putText(
            annotated_image,
            prediction_label,
            (x_min, label_y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # Step 14: Save the annotated image for manual inspection.
    output_image_name = (
        "image_"
        + str(image_number).zfill(2)
        + "_"
        + os.path.basename(image_path)
    )

    output_image_path = os.path.join(
        annotated_image_folder,
        output_image_name
    )

    cv2.imwrite(output_image_path, annotated_image)

    print("Detections:", number_of_detections)
    print("Annotated image saved:", output_image_name)

# Step 15: Save all prediction records to a CSV file.
csv_field_names = [
    "image_name",
    "detection_number",
    "predicted_class",
    "yolo_confidence",
    "cnn_confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max"
]

with open(
    csv_output_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=csv_field_names
    )

    csv_writer.writeheader()
    csv_writer.writerows(all_prediction_records)

# Step 16: Display the final batch testing summary.
print("\n========== BATCH TEST SUMMARY ==========")
print("Images selected:", number_of_images_to_test)
print("Images with detections:", images_with_detections)
print("Total detected objects:", total_detected_objects)

if number_of_images_to_test > 0:
    print(
        "Percentage of images with detections:",
        round(
            images_with_detections / number_of_images_to_test * 100,
            2
        ),
        "%"
    )

print("\nCSV results saved to:")
print(csv_output_path)

print("\nAnnotated images saved to:")
print(annotated_image_folder)

print("\nBatch testing finished.")

