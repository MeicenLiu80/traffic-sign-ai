// This file handles image selection, drag-and-drop, and the prediction request to Flask.
document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("upload-form");
    const imageInput = document.getElementById("image-input");
    const sampleButtons = document.querySelectorAll(".sample-option");
    const dropZone = document.getElementById("drop-zone");
    const selectedFile = document.getElementById("selected-file");
    const analyzeButton = document.getElementById("analyze-button");
    const buttonText = document.getElementById("button-text");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const loadingState = document.getElementById("loading-state");
    const errorMessage = document.getElementById("error-message");
    const resultContent = document.getElementById("result-content");
    const resultImage = document.getElementById("result-image");
    const resultSummary = document.getElementById("result-summary");
    const predictionList = document.getElementById("prediction-list");

    let selectedImage = null;

    // Update the selected file and display its name before analysis.
    function chooseImage(file) {
        if (!file) {
            return;
        }

        const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!allowedTypes.includes(file.type)) {
            showError("Please choose a JPG, PNG, WEBP, or BMP image.");
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            showError("The image is too large. Please choose an image under 10 MB.");
            return;
        }

        selectedImage = file;
        selectedFile.hidden = false;
        selectedFile.textContent = "Selected: " + file.name;
        errorMessage.hidden = true;
        resultContent.hidden = true;
        resultPlaceholder.hidden = false;
    }

    // Display an error message and hide the loading indicator.
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.hidden = false;
        loadingState.hidden = true;
        analyzeButton.disabled = false;
        buttonText.textContent = "Analyze traffic signs";
    }

    // Handle image selection through the file picker.
    imageInput.addEventListener("change", function () {
        if (imageInput.files.length > 0) {
            chooseImage(imageInput.files[0]);
        }
    });

    // Load a bundled sample image and send it through the same existing prediction form.
    sampleButtons.forEach(function (sampleButton) {
        sampleButton.addEventListener("click", async function () {
            const sampleUrl = sampleButton.dataset.sampleUrl;
            const sampleName = sampleButton.dataset.sampleName || "canadian-road-sample.webp";

            sampleButton.disabled = true;

            try {
                const sampleResponse = await fetch(sampleUrl);

                if (!sampleResponse.ok) {
                    throw new Error("The selected example image could not be loaded.");
                }

                const sampleBlob = await sampleResponse.blob();
                const sampleFile = new File(
                    [sampleBlob],
                    sampleName,
                    { type: sampleBlob.type || "image/webp" }
                );

                chooseImage(sampleFile);
                uploadForm.requestSubmit();
            } catch (error) {
                showError(error.message || "The sample image could not be used.");
                sampleButton.disabled = false;
            }
        });
    });

    // Handle drag-and-drop upload interactions.
    ["dragenter", "dragover"].forEach(function (eventName) {
        dropZone.addEventListener(eventName, function (event) {
            event.preventDefault();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
        dropZone.addEventListener(eventName, function (event) {
            event.preventDefault();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", function (event) {
        if (event.dataTransfer.files.length > 0) {
            chooseImage(event.dataTransfer.files[0]);
        }
    });

    // Convert a prediction result into a readable confidence percentage.
    function percentage(value) {
        return (Number(value) * 100).toFixed(1) + "%";
    }

    // Render each detected sign as a small result card.
    function renderPredictions(predictions) {
        predictionList.innerHTML = "";

        predictions.forEach(function (prediction) {
            const item = document.createElement("div");
            item.className = "prediction-item";

            const information = document.createElement("div");
            information.className = "prediction-information";

            const name = document.createElement("div");
            name.className = "prediction-name";
            name.textContent = prediction.class_name;

            const meta = document.createElement("div");
            meta.className = "prediction-meta";
            meta.textContent =
                "YOLO detection: " + percentage(prediction.yolo_confidence) +
                " · CNN classification: " + percentage(prediction.cnn_confidence);

            information.appendChild(name);
            information.appendChild(meta);

            const chip = document.createElement("span");
            chip.className = "prediction-chip";
            chip.textContent = "Sign " + prediction.detection_number;

            item.appendChild(information);
            item.appendChild(chip);
            predictionList.appendChild(item);
        });
    }

    // Send the selected image to the Flask backend and display the response.
    uploadForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        if (!selectedImage) {
            showError("Please choose an image before starting the analysis.");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedImage);

        analyzeButton.disabled = true;
        sampleButtons.forEach(function (sampleButton) {
            sampleButton.disabled = true;
        });
        buttonText.textContent = "Analyzing...";
        resultPlaceholder.hidden = true;
        errorMessage.hidden = true;
        resultContent.hidden = true;
        loadingState.hidden = false;

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "The prediction request failed.");
            }

            resultImage.src = data.result_image + "?t=" + Date.now();
            resultSummary.textContent =
                data.detections_count + " detection(s) · " + data.message;

            renderPredictions(data.predictions);

            loadingState.hidden = true;
            resultContent.hidden = false;
            document.getElementById("recognition-results").scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        } catch (error) {
            showError(error.message || "Something went wrong while analyzing the image.");
        } finally {
            analyzeButton.disabled = false;
            sampleButtons.forEach(function (sampleButton) {
                sampleButton.disabled = false;
            });
            buttonText.textContent = "Analyze traffic signs";
        }
    });
});
