# This script combines the trained YOLO detector and the trained CNN classifier.
# YOLO detects traffic sign locations, and CNN predicts the class of each detected sign.
# The script uses the existing model weights and does not perform any model training.
# It saves an annotated image and prints the detection and classification results.

import os
import random
import json
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

# Step 2: Check whether all required files and folders exist.
required_paths = [
    yolo_model_path,
    cnn_model_path,
    class_mapping_path,
    validation_image_folder
]

for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError(
            "A required file or folder was not found: " + required_path
        )

print("All required model files and folders were found.")

# Step 3: Select the GPU if CUDA is available.
if torch.cuda.is_available():
    selected_device = 0
    torch_device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    selected_device = "cpu"
    torch_device = torch.device("cpu")
    print("CUDA is unavailable. Using CPU.")

# Step 4: Define the CNN architecture to match the saved model weights.
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

# Step 5: Load the saved class mapping.
with open(class_mapping_path, "r", encoding="utf-8") as mapping_file:
    class_to_index = json.load(mapping_file)

index_to_class = {
    int(class_index): class_name
    for class_name, class_index in class_to_index.items()
}

number_of_classes = len(class_to_index)

print("Number of classification classes:", number_of_classes)

# Step 6: Load the saved YOLO detector.
print("\nLoading YOLO detector...")

yolo_model = YOLO(yolo_model_path)

print("YOLO detector loaded successfully.")

# Step 7: Load the saved CNN classifier.
print("\nLoading CNN classifier...")

cnn_model = TrafficSignCNN(
    number_of_classes=number_of_classes
)

cnn_weights = torch.load(
    cnn_model_path,
    map_location="cpu",
    weights_only=True
)

cnn_model.load_state_dict(cnn_weights, strict=True)
cnn_model = cnn_model.to(torch_device)
cnn_model.eval()

print("CNN classifier loaded successfully.")

# Step 8: Define the image preprocessing for the CNN.
# This uses the same preliminary preprocessing that worked in our earlier test.
cnn_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Step 9: Find original validation images and select one image.
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

validation_image_paths = []

for file_name in os.listdir(validation_image_folder):
    if file_name.lower().endswith(image_extensions):
        validation_image_paths.append(
            os.path.join(validation_image_folder, file_name)
        )

if len(validation_image_paths) == 0:
    raise FileNotFoundError(
        "No validation images were found in the validation folder."
    )

random.seed(10)

selected_image_path = random.choice(validation_image_paths)

print("\nSelected input image:")
print(selected_image_path)

# Step 10: Open the original image in RGB format.
with Image.open(selected_image_path) as image_file:
    original_image = image_file.convert("RGB")

# Step 11: Run YOLO detection on the selected image.
print("\nRunning YOLO detection...")

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

first_result = detection_results[0]

if first_result.boxes is not None:
    detected_boxes = first_result.boxes
else:
    detected_boxes = []

print("YOLO detection completed.")
print("Number of detected objects:", len(detected_boxes))

# Step 12: Create a copy of the original image for drawing the results.
annotated_image = cv2.cvtColor(
    cv2.imread(selected_image_path),
    cv2.COLOR_BGR2RGB
)

if annotated_image is None:
    raise RuntimeError("The selected input image could not be read.")

# Step 13: Crop each detected object and classify it with the CNN.
prediction_results = []

for detection_number, detected_box in enumerate(
    detected_boxes,
    start=1
):

    # Read the bounding box coordinates from the YOLO result.
    box_coordinates = detected_box.xyxy[0].cpu().tolist()

    x_min = int(box_coordinates[0])
    y_min = int(box_coordinates[1])
    x_max = int(box_coordinates[2])
    y_max = int(box_coordinates[3])

    # Make sure the coordinates remain inside the original image.
    image_height, image_width = annotated_image.shape[:2]

    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(x_min + 1, min(x_max, image_width))
    y_max = max(y_min + 1, min(y_max, image_height))

    # Crop the detected traffic sign from the original RGB image.
    cropped_array = annotated_image[y_min:y_max, x_min:x_max]

    if cropped_array.size == 0:
        print("Skipping an empty crop.")
        continue

    cropped_sign = Image.fromarray(cropped_array).convert("RGB")

    # Convert the cropped sign into a CNN input tensor.
    cnn_input = cnn_image_transform(cropped_sign).unsqueeze(0)
    cnn_input = cnn_input.to(torch_device)

    # Run CNN classification without calculating gradients.
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

    classification_confidence = float(class_confidence.item())
    detection_confidence = float(detected_box.conf[0].item())

    # Save the prediction information for the terminal output.
    prediction_information = {
        "detection_number": detection_number,
        "class_name": predicted_class_name,
        "classification_confidence": classification_confidence,
        "detection_confidence": detection_confidence,
        "bounding_box": (x_min, y_min, x_max, y_max)
    }

    prediction_results.append(prediction_information)

    # Draw the bounding box around the detected traffic sign.
    cv2.rectangle(
        annotated_image,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    # Prepare a readable label containing the predicted class and confidence.
    prediction_label = (
        predicted_class_name
        + " | YOLO "
        + str(round(detection_confidence * 100, 1))
        + "%"
        + " | CNN "
        + str(round(classification_confidence * 100, 1))
        + "%"
    )

    # Draw the label above the bounding box when possible.
    label_y_position = max(20, y_min - 8)

    cv2.putText(
        annotated_image,
        prediction_label,
        (x_min, label_y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

# Step 14: Create the output folder and save the annotated image.
output_folder = os.path.join(
    project_folder,
    "YOLO_Detection_Results",
    "integrated_results"
)

os.makedirs(output_folder, exist_ok=True)

output_image_path = os.path.join(
    output_folder,
    "integrated_traffic_sign_result.jpg"
)

# OpenCV expects BGR channel order when saving an image.
image_to_save = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_RGB2BGR
)

image_saved = cv2.imwrite(
    output_image_path,
    image_to_save
)

if image_saved:
    print("\nAnnotated result image saved:")
    print(output_image_path)
else:
    print("\nWarning: The annotated image could not be saved.")

# Step 15: Print the final detection and classification results.
print("\n========== INTEGRATED PREDICTION RESULTS ==========")

if len(prediction_results) == 0:
    print("No traffic signs were detected in this image.")

else:
    for prediction in prediction_results:

        print("\nDetection number:", prediction["detection_number"])
        print("Predicted traffic sign:", prediction["class_name"])

        print(
            "YOLO detection confidence:",
            round(
                prediction["detection_confidence"] * 100,
                2
            ),
            "%"
        )

        print(
            "CNN classification confidence:",
            round(
                prediction["classification_confidence"] * 100,
                2
            ),
            "%"
        )

        print(
            "Bounding box:",
            prediction["bounding_box"]
        )

print("\nIntegrated traffic sign recognition finished.")

