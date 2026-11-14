# Traffic Sign Detection and Classification Using Deep Learning

## 1. Introduction

Traffic signs provide important information to drivers, making traffic sign recognition a key component of autonomous driving technology (Ertler et al., 2020). This project uses YOLO11n to detect traffic signs in road images and a convolutional neural network (CNN) to classify the detected signs.

In addition to model development, a web-based demonstration system was implemented and deployed at [AutoVisionLab](https://autovisionlab.online). Users can upload road images and view the detected traffic signs, predicted classes, and confidence scores.

## 2. Related Work

YOLO (You Only Look Once) is a single-stage object detection method that uses a neural network to directly predict bounding boxes and class probabilities from input images. This approach provides an efficient detection process and is suitable for applications that require fast image processing (Redmon et al., 2016).

In this project, YOLO11n was selected for object detection. All traffic signs are treated as a single object category named `traffic_sign`, allowing the model to focus on locating traffic signs rather than identifying their specific types (Jocher & Qiu, 2024).

## 3. Methods

### 3.1 Dataset Preparation

The Mapillary Traffic Sign Dataset (MTSD) was used for model development. For the YOLO11n model, full road images and their corresponding bounding box annotations were used.

For the classification task, 326 traffic sign classes were selected. Classes with fewer than 30 images were excluded to reduce the impact of insufficient training samples. After data preparation, the classification dataset contained 60,162 training images and 8,736 validation images.

All classification images were resized to 64 × 64 pixels to provide consistent input dimensions for the CNN model.

### 3.2 Experimental Setup

The project was implemented using Python and deep learning frameworks for model development and training. The trained model checkpoints were integrated into a web application called AutoVisionLab ([https://autovisionlab.online](https://autovisionlab.online)), which allows users to upload road images and view the recognition results.

The experiments included training a YOLO11n object detection model and comparing two CNN classification models: a baseline model and a model trained with data augmentation.

CNN performance was evaluated using validation accuracy, while YOLO11n performance was evaluated using precision, recall, and mean Average Precision (mAP).

### 3.3 Object Detection Model

YOLO11n was used for traffic sign detection, with all traffic signs classified into a single category named `traffic_sign`.

The training configuration used an image size of 640 × 640 pixels and a maximum of 50 training epochs. The model was trained to locate traffic signs in full road images and generate bounding boxes around detected objects.

### 3.4 Image Classification Model

A CNN was developed to classify the detected traffic signs into 326 different categories. The input images contain three color channels (RGB).

The network consists of three convolutional stages with 32, 64, and 128 output channels, respectively. Each stage uses the ReLU activation function followed by 2 × 2 max pooling to extract visual features and reduce the spatial dimensions of the feature maps.

After the convolutional layers, the extracted feature maps are flattened and passed into a fully connected layer with 256 neurons. A dropout rate of 0.3 is used to reduce overfitting. The final output layer contains 326 neurons, corresponding to the selected traffic sign classes.

The model was trained for 20 epochs using the Adam optimizer with a learning rate of 0.001 and a batch size of 64. Cross-entropy loss was used as the classification loss function. The random seed was set to 42 to improve experimental reproducibility.

Data augmentation was also applied to the training images to improve the model's ability to recognize traffic signs under different conditions. The augmentation techniques included random rotation (±10°), translation (up to 5%), scaling (0.95–1.05), and color jitter.

## 4. Results and Discussion

### 4.1 Experimental Results

The baseline CNN achieved its best validation accuracy of 90.66% at epoch 19. After applying data augmentation, the best validation accuracy improved to 92.91% at epoch 17. However, the augmented model required a longer training time.

The YOLO11n object detection model achieved a precision of 0.73038 and a recall of 0.56097. It also achieved an mAP@50 of 0.60818 and an mAP@50–95 of 0.40564.

### 4.2 Performance Analysis

The experimental results show that data augmentation improved the classification performance of the CNN. The validation accuracy increased from 90.66% to 92.91%, representing an improvement of 2.25 percentage points. However, data augmentation also increased the training time, creating a trade-off between model performance and computational cost.

For YOLO11n, the precision score was higher than the recall score. This indicates that the model was relatively more successful at making correct detections than at finding all traffic signs in the images. In other words, some traffic signs were still missed during the detection process.

The overall performance of the recognition system depends on both models. If a traffic sign is missed during the object detection stage, the CNN cannot classify it. Therefore, improving the detection model's recall could also improve the performance of the complete recognition system.

### 4.3 Limitations

Although the classification results are promising, the project still has several limitations.

First, the dataset was filtered, and all classes with fewer than 30 training images were excluded. As a result, the classification model cannot recognize every traffic sign category in the original dataset.

Second, the YOLO11n model achieved a validation recall of 0.56097, indicating that some traffic signs were not detected.

Finally, the reported performance results are based on the validation dataset rather than an independent test set. Therefore, the results may not fully represent the model's performance on unseen real-world road images.

## 5. Conclusion and Future Work

This project developed a traffic sign recognition system by combining YOLO11n object detection with CNN image classification. The experimental results showed that data augmentation improved the classification accuracy of the CNN, while the YOLO11n model still missed some traffic signs.

These results demonstrate that combining object detection and image classification provides a practical approach to traffic sign recognition. However, the overall system performance still depends heavily on the accuracy of the detection stage.

Several improvements can be considered for future work. First, the object detection model could be further improved, with particular attention to increasing recall and reducing missed detections. Second, other CNN architectures could be explored to achieve a better balance between classification accuracy, model size, and training efficiency. In addition, the dataset could be expanded to include more diverse traffic signs and road conditions.

Through this project, I gained practical experience in applying deep learning techniques to a real-world computer vision problem. I worked on image dataset preparation, neural network training, model evaluation, and the integration of different deep learning models into a complete recognition system. I also gained a better understanding of how object detection and image classification models can work together to solve a practical engineering problem.

## 6. References

Ertler, C., Mislej, J., Ollmann, T., Porzi, L., Neuhold, G., & Kuang, Y. (2020). The Mapillary Traffic Sign Dataset for detection and classification on a global scale. *Computer Vision – ECCV 2020*, 68–84. https://doi.org/10.1007/978-3-030-58592-1_5

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 779–788. https://doi.org/10.1109/CVPR.2016.91

Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* (Version 11.0.0) [Computer software]. GitHub. https://github.com/ultralytics/ultralytics

## Acknowledgment of Generative AI Use

Generative AI tools, particularly ChatGPT, were used to assist with model design, code development, and troubleshooting. However, the project was not entirely completed by AI. I actively participated in dataset preparation, model training, experimentation, and performance evaluation. I also reviewed AI-generated suggestions and take responsibility for the accuracy of the final results and content.
