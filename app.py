# This file runs the Flask website and connects the web page to the existing YOLO and CNN models.
# It does not train either model. It loads the saved checkpoints once when the server starts.

import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Step 1: Define the main folders used by this website.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

YOLO_MODEL_PATH = MODEL_DIR / "yolo_best.pt"
CNN_MODEL_PATH = MODEL_DIR / "cnn_augmentation_best.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "final_class_to_index.json"

# Step 2: Define upload and inference settings.
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_UPLOAD_SIZE_MB = 10
YOLO_IMAGE_SIZE = 320
YOLO_CONFIDENCE_THRESHOLD = 0.25
# This part defines the minimum CNN confidence for showing a classification.
CNN_CLASSIFICATION_THRESHOLD = 0.50
MAX_DETECTIONS = 30

# Step 3: Create required folders if they do not exist yet.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Step 4: Create the Flask application.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024


# Step 5: Define the CNN architecture so it matches the saved checkpoint.
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
            # This dropout probability matches the recovered CNN training experiment.
            nn.Dropout(p=0.3),
            nn.Linear(256, number_of_classes)
        )

    def forward(self, input_images):
        extracted_features = self.convolutional_layers(input_images)
        output_scores = self.classification_layers(extracted_features)
        return output_scores


# Step 6: Select a computing device and prepare model variables.
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL = None
CNN_MODEL = None
INDEX_TO_CLASS = {}
MODEL_LOAD_ERROR = None

# Step 7: Define the CNN image preprocessing used in our earlier working test.
# The training script was not found, so this is the tested preliminary preprocessing.
CNN_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# Step 8: Load both existing models and the class mapping.
def load_models():
    global YOLO_MODEL, CNN_MODEL, INDEX_TO_CLASS, MODEL_LOAD_ERROR

    missing_paths = [
        str(path)
        for path in [YOLO_MODEL_PATH, CNN_MODEL_PATH, CLASS_MAPPING_PATH]
        if not path.is_file()
    ]

    if missing_paths:
        MODEL_LOAD_ERROR = (
            "Required model files are missing. Place the saved files in the "
            "'models' folder. Missing: " + ", ".join(missing_paths)
        )
        print(MODEL_LOAD_ERROR)
        return

    try:
        with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as mapping_file:
            class_to_index = json.load(mapping_file)

        INDEX_TO_CLASS = {
            int(class_index): class_name
            for class_name, class_index in class_to_index.items()
        }

        YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))

        cnn_weights = torch.load(
            CNN_MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )

        CNN_MODEL = TrafficSignCNN(number_of_classes=len(INDEX_TO_CLASS))
        CNN_MODEL.load_state_dict(cnn_weights, strict=True)
        CNN_MODEL = CNN_MODEL.to(TORCH_DEVICE)
        CNN_MODEL.eval()

        MODEL_LOAD_ERROR = None
        print("Traffic Sign AI models loaded successfully.")
        print("Using device:", TORCH_DEVICE)
        print("CNN classes:", len(INDEX_TO_CLASS))

    except Exception as error:
        MODEL_LOAD_ERROR = "Model loading failed: " + str(error)
        print(MODEL_LOAD_ERROR)


# Step 9: Check whether an uploaded filename has an allowed image extension.
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Step 10: Run YOLO detection followed by CNN classification on every detected crop.
def recognize_traffic_signs(image_path, result_image_path):
    if YOLO_MODEL is None or CNN_MODEL is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Models are not ready.")

    original_bgr = cv2.imread(str(image_path))
    if original_bgr is None:
        raise ValueError("The uploaded file could not be read as an image.")

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    image_height, image_width = original_bgr.shape[:2]
    annotated_bgr = original_bgr.copy()

    with torch.inference_mode():
        detection_results = YOLO_MODEL.predict(
            source=str(image_path),
            imgsz=YOLO_IMAGE_SIZE,
            batch=1,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )

    first_result = detection_results[0]
    detected_boxes = first_result.boxes if first_result.boxes is not None else []
    predictions = []

    for detected_box in detected_boxes:
        coordinates = detected_box.xyxy[0].cpu().tolist()

        x_min = max(0, min(int(coordinates[0]), image_width - 1))
        y_min = max(0, min(int(coordinates[1]), image_height - 1))
        x_max = max(x_min + 1, min(int(coordinates[2]), image_width))
        y_max = max(y_min + 1, min(int(coordinates[3]), image_height))

        cropped_array = original_rgb[y_min:y_max, x_min:x_max]
        if cropped_array.size == 0:
            continue

        cropped_sign = Image.fromarray(cropped_array).convert("RGB")
        cnn_input = CNN_IMAGE_TRANSFORM(cropped_sign).unsqueeze(0).to(TORCH_DEVICE)

        with torch.inference_mode():
            scores = CNN_MODEL(cnn_input)
            probabilities = torch.softmax(scores, dim=1)
            class_confidence, predicted_index = torch.max(probabilities, dim=1)

        predicted_class_index = int(predicted_index.item())
        predicted_class_name = INDEX_TO_CLASS[predicted_class_index]
        cnn_confidence = float(class_confidence.item())
        yolo_confidence = float(detected_box.conf[0].item())

        # This part skips classifications that do not reach the 50% requirement.
        if cnn_confidence < CNN_CLASSIFICATION_THRESHOLD:
            continue

        predictions.append({
            # This part keeps the displayed sign numbers consecutive after filtering.
            "detection_number": len(predictions) + 1,
            "class_name": predicted_class_name,
            "yolo_confidence": round(yolo_confidence, 4),
            "cnn_confidence": round(cnn_confidence, 4),
            "box": [x_min, y_min, x_max, y_max]
        })

        # Draw a compact numbered marker so multiple predictions do not create overlapping labels.
        cv2.rectangle(annotated_bgr, (x_min, y_min), (x_max, y_max), (50, 220, 110), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "#" + str(len(predictions))
        font_scale = 0.50
        thickness = 1
        label_size, _label_baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_width = label_size[0]
        label_height = label_size[1]
        label_padding = 5
        badge_width = label_width + (label_padding * 2)
        badge_height = label_height + (label_padding * 2)

        # Keep the small number tag inside the box when possible, or just above a very short box.
        if (y_max - y_min) >= badge_height:
            label_top = y_min
        else:
            label_top = max(0, y_min - badge_height)

        label_left = min(x_min, max(0, image_width - badge_width))
        label_right = min(image_width - 1, label_left + badge_width)
        label_bottom = min(image_height - 1, label_top + badge_height)

        # Draw a solid background behind the short marker to keep it readable on any road scene.
        cv2.rectangle(
            annotated_bgr,
            (label_left, label_top),
            (label_right, label_bottom),
            (50, 220, 110),
            -1
        )

        label_x = label_left + label_padding
        label_y = min(image_height - 1, label_top + label_padding + label_height)
        cv2.putText(
            annotated_bgr, label, (label_x, label_y), font,
            font_scale, (20, 45, 30), thickness, cv2.LINE_AA
        )

    if not cv2.imwrite(str(result_image_path), annotated_bgr):
        raise RuntimeError("The result image could not be saved.")

    return predictions, image_width, image_height, len(detected_boxes)


# Step 11: Render the website's main page.
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=(YOLO_MODEL is not None and CNN_MODEL is not None),
        model_error=MODEL_LOAD_ERROR,
        device_name=("CUDA GPU" if torch.cuda.is_available() else "CPU")
    )


# Step 12: Receive an uploaded image, run inference, and return JSON results.
@app.route("/api/predict", methods=["POST"])
def predict():
    if YOLO_MODEL is None or CNN_MODEL is None:
        return jsonify({
            "error": MODEL_LOAD_ERROR or "Models are not ready. Check the models folder."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "Please choose an image first."}), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Please choose an image first."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "error": "Unsupported file type. Please upload JPG, PNG, WEBP, or BMP."
        }), 400

    safe_filename = secure_filename(uploaded_file.filename)
    unique_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{unique_id}_{safe_filename}"
    result_filename = f"{unique_id}_result.jpg"
    result_path = RESULT_DIR / result_filename

    try:
        uploaded_file.save(upload_path)

        # Verify that OpenCV can decode the upload before running the models.
        test_image = cv2.imread(str(upload_path))
        if test_image is None:
            return jsonify({"error": "The uploaded file is not a readable image."}), 400

        predictions, image_width, image_height, detected_count = recognize_traffic_signs(
            upload_path,
            result_path
        )

        return jsonify({
            "result_image": url_for("static", filename=f"results/{result_filename}"),
            "image_width": image_width,
            "image_height": image_height,
            "detections_count": len(predictions),
            "predictions": predictions,
            "message": (
                "Recognition completed."
                if predictions
                else (
                    "No classifications met the 50% CNN confidence threshold."
                    if detected_count > 0
                    else "No traffic signs were detected in this image."
                )
            )
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed: " + str(error)}), 500

    finally:
        # Remove the uploaded copy after inference. The annotated result remains available.
        try:
            if upload_path.exists():
                upload_path.unlink()
        except OSError:
            pass


# Step 13: Show a clear message for uploads that exceed the configured size limit.
@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": f"Image is too large. Please upload an image under {MAX_UPLOAD_SIZE_MB} MB."
    }), 413


# Step 14: Load the saved models when the website starts.
load_models()

# Step 15: Run the local development server.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


