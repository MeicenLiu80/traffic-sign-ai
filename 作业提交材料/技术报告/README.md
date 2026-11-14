# Traffic Sign Detection and Classification Using Deep Learning

## 1. Introduction

Traffic signs provide important information to drivers, making traffic sign recognition a key component of autonomous driving technology (Ertler et al., 2020).

This project uses YOLO11n to detect traffic signs in road images and a convolutional neural network (CNN) to classify the detected signs. In addition to model development, a demo system called [AutoVisionLab](https://autovisionlab.online) was implemented. Users can upload images and view the detected traffic signs, predicted classes, and confidence scores.

## 2. Related Work

YOLO (You Only Look Once) is a single-stage object detection method that uses a neural network to directly predict bounding boxes and class probabilities from input images. This method provides an efficient detection pipeline suitable for applications requiring fast image processing (Redmon et al., 2016).

YOLO11n was used for object detection, with all traffic signs classified into a single category named `traffic_sign` (Jocher & Qiu, 2024).

## 3. Methods

### 3.1 Dataset Preparation

The Mapillary Traffic Sign Dataset (MTSD) was used for training data (Ertler et al., 2020). For the YOLO11n model, full road images and their corresponding bounding box annotations were used.

The classification task included 326 selected classes. Classes with fewer than 30 training images were excluded. After data preparation, the dataset contained 60,162 training images and 8,736 validation images.

All classification images were resized to 64 × 64 pixels to provide consistent input dimensions for the CNN model.

### 3.2 Experimental Setup

The project was implemented using Python and deep learning frameworks for model development and training. The trained model checkpoints were then integrated into a web application called [AutoVisionLab](https://autovisionlab.online), which allows users to upload road images and view the recognition results.

The experiments included training the YOLO11n model and comparing two CNN models: a baseline model and a model trained with data augmentation.

CNN performance was evaluated using validation accuracy, while YOLO11n performance was evaluated using precision, recall, and mean Average Precision (mAP).

### 3.3 Object Detection Model

YOLO11n was used for object detection, with all traffic signs classified into a single category named `traffic_sign`.

The training configuration used an image size of 640 pixels and a maximum of 50 training epochs.

### 3.4 Image Classification Model

A CNN was used to classify the detected traffic signs. The input images contain three RGB color channels.

The network consists of three convolutional stages with 32, 64, and 128 output channels, respectively. Each stage uses the ReLU activation function and 2 × 2 max pooling.

After the convolutional layers, the feature maps are flattened and fed into a fully connected layer with 256 neurons. A dropout rate of 0.5 is used to reduce overfitting. The output layer contains 326 neurons, corresponding to the selected traffic sign classes.

The model was trained for 20 epochs using the Adam optimizer with a learning rate of 0.001 and a batch size of 64. The classification loss function was cross-entropy loss, and the random seed was set to 42 to ensure reproducibility.

Data augmentation was applied to the CNN training images, including random rotation (±10°), translation (5%), scaling (0.95–1.05), and color jitter.

## 4. Results and Discussion

### 4.1 Results

The baseline CNN achieved its best validation accuracy of 90.66% at epoch 19. After applying data augmentation, the best validation accuracy improved to 92.91% at epoch 17. However, the augmented model required a longer training time.

The YOLO11n object detection model achieved a precision of 0.73038 and a recall of 0.56097. It also achieved an mAP@50 of 0.60818 and an mAP@50–95 of 0.40564.

### 4.2 Performance Analysis

The experimental results show that data augmentation has a positive impact on the classification performance of the CNN, but it also significantly increases training time, creating a trade-off between performance and computational cost.

YOLO11n performs better at correctly detecting traffic signs than at finding all traffic signs in an image. This means that although many detected signs are correct, the model still misses some traffic signs.

A complete recognition system depends on the performance of both models. If a traffic sign is missed during the detection stage, the CNN cannot classify it.

### 4.3 Limitations

Although the classification results are promising, there are still several limitations.

First, the dataset was filtered, and all classes with fewer than 30 training images were excluded. As a result, the classification model cannot recognize all traffic sign classes.

Second, the validation recall of YOLO11n is 0.56097, indicating that some traffic signs were not detected.

In addition, the performance results are based on the validation dataset rather than an independent test set.

## 5. Conclusion and Future Work

The experimental results show that data augmentation improved the classification performance of the CNN. However, the YOLO11n model still missed some traffic signs.

These results indicate that combining object detection with image classification can be effectively applied to traffic sign recognition. However, detection performance still needs to be improved.

Several improvements can be considered. First, the object detection model can be improved, with a focus on increasing recall. Second, other CNN architectures can be explored to balance model size and training efficiency. In addition, the dataset can be expanded to include more diverse traffic signs.

Through this project, I gained practical experience in applying deep learning techniques to real-world problems. I practiced preparing image datasets, training neural network models, and evaluating their performance using different metrics. I also gained a better understanding of how different deep learning models work together in a complete recognition system.

## 6. References

Ertler, C., Mislej, J., Ollmann, T., Porzi, L., Neuhold, G., & Kuang, Y. (2020). The Mapillary Traffic Sign Dataset for detection and classification on a global scale. In *Computer Vision – ECCV 2020* (pp. 68–84). https://doi.org/10.1007/978-3-030-58592-1_5

Jocher, G., & Qiu, J. (2024). *Ultralytics YOLO11* (Version 11.0.0) [Computer software]. GitHub. https://github.com/ultralytics/ultralytics

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 779–788). https://doi.org/10.1109/CVPR.2016.91

## 7. Acknowledgment of Generative AI Use

Generative AI tools, particularly ChatGPT, were used to assist with model design, code development, and troubleshooting. However, the project was not entirely completed by AI.

I actively participated in dataset preparation, model training, experimentation, and performance evaluation. I also reviewed AI-generated suggestions and take responsibility for the accuracy of the final results and content.
