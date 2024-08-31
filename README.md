# Detection-and-Classification-of-Iceberg-and-Sea-Ice-Patterns-in-Satellite-Imagery

## Overview

This project addresses the critical challenge of classifying sea ice and icebergs using satellite imagery and advanced deep learning techniques. The goal is to enhance the accuracy and efficiency of monitoring polar regions, contributing to climate science and environmental management efforts. Leveraging Convolutional Neural Networks (CNNs) and the power of high-resolution satellite data, this project provides a robust solution to the problem of distinguishing between sea ice and icebergs—two crucial components of the Earth's cryosphere.

## Output
![class](https://github.com/user-attachments/assets/939a9da5-3612-4468-868c-0d37c4dce940)


## Motivation and Ideas

The motivation behind this project stems from a deep concern for the accelerating impacts of climate change, particularly in polar regions where the effects are most pronounced. The melting of sea ice and the calving of icebergs are not only visual indicators of global warming but also have far-reaching implications for sea level rise, ocean circulation, and weather patterns across the globe.

### Key Ideas Driving the Project:

1. **Harnessing Deep Learning for Environmental Monitoring**: Recognizing the power of deep learning, especially CNNs, to extract complex patterns from visual data, I was inspired to apply these techniques to the classification of sea ice and icebergs. This approach promises to automate the monitoring process, making it faster, more accurate, and scalable.

2. **Addressing Data Imbalance**: A major challenge in environmental datasets is class imbalance, where certain phenomena (like icebergs) are less frequently observed compared to others (like sea ice). I was motivated to explore techniques such as class-aware sampling and pseudo-labeling to ensure that the model does not become biased towards more common classes, thereby improving its ability to generalize across different conditions.

3. **Integrating Real-Time Capabilities**: With climate change accelerating, the ability to monitor polar regions in real-time has become increasingly important. This project is a step towards creating a system that can process satellite data in real-time, providing immediate insights into the state of the cryosphere and enabling quicker responses to environmental changes.

4. **Contributing to Global Climate Research**: The idea that this project could contribute to the broader scientific understanding of climate dynamics was a significant motivator. By improving the tools available for monitoring the polar regions, I hope to provide valuable data that can inform both scientific research and policy decisions.

## Background

### The Cryosphere and Climate Change

The cryosphere, comprising all of Earth's frozen water bodies, plays a pivotal role in regulating the global climate. Sea ice and icebergs are not only critical indicators of climate change but also influence oceanic and atmospheric circulation patterns. Changes in the extent and thickness of sea ice can significantly impact global climate systems, making it essential to monitor these changes accurately and in real time.

Sea ice serves as a reflective barrier that limits the amount of solar radiation absorbed by the Earth’s surface, thus regulating temperature. The melting of sea ice reduces this albedo effect, leading to further warming—a positive feedback loop that accelerates climate change. Similarly, icebergs, which calve from glaciers and ice sheets, contribute to sea level rise and can disrupt marine ecosystems.

### Challenges in Monitoring the Cryosphere

Monitoring sea ice and icebergs using satellite imagery presents several challenges:

1. **Variability in Visual Characteristics**: Sea ice and icebergs exhibit significant variability in their visual characteristics due to differences in texture, reflectance, and size. These variations can be caused by factors such as age, thickness, snow cover, and melting conditions.
  
2. **Class Imbalance**: In many datasets, the number of sea ice images may vastly outnumber iceberg images (or vice versa), leading to class imbalance. This can skew the learning process of neural networks, making them biased toward the majority class.

3. **Environmental Noise**: Satellite images are often affected by environmental noise, such as clouds, fog, or varying lighting conditions, which can obscure the features of sea ice and icebergs.

### Deep Learning in Environmental Monitoring

Deep learning, and specifically Convolutional Neural Networks (CNNs), have revolutionized the field of image classification. CNNs are particularly effective for tasks involving complex spatial hierarchies, such as those found in satellite images. A CNN automatically learns to detect relevant features (like edges, textures, and patterns) from the input images, making it an ideal tool for distinguishing between sea ice and icebergs.

### Theoretical Concepts

#### Convolutional Neural Networks (CNNs)

CNNs are a class of deep neural networks designed to process and classify visual data. The architecture of a CNN typically includes several layers:

1. **Convolutional Layers**: These layers apply a series of filters to the input images to extract features. The convolution operation detects patterns such as edges and textures by sliding a filter (or kernel) over the image and computing dot products at each location.

2. **Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps while retaining the most important information. Max pooling, the most common pooling technique, selects the maximum value from each region of the feature map.

3. **Fully Connected Layers**: After convolutional and pooling layers have extracted features from the images, fully connected layers use these features to perform the final classification. Each neuron in these layers is connected to every neuron in the previous layer, enabling the network to combine features to make decisions.

4. **Dropout Layers**: Dropout is a regularization technique used to prevent overfitting by randomly "dropping out" a fraction of neurons during training. This forces the network to learn more robust features that are not reliant on specific neurons.

#### Binary Classification with CNNs

In this project, CNNs are used for binary classification, where the goal is to classify images as either sea ice or iceberg. The output layer of the CNN has a single neuron with a sigmoid activation function, which outputs a probability value between 0 and 1. This probability is then thresholded to make a binary decision (e.g., sea ice vs. iceberg).

The network is trained using the **binary crossentropy loss function**, which measures the difference between the predicted probability and the actual label. The network’s weights are updated iteratively to minimize this loss, improving the model's ability to correctly classify new images.

### Handling Class Imbalance

Class imbalance is a common issue in binary classification tasks. When one class significantly outnumbers the other, the model may become biased, leading to poor generalization on the minority class. To address this, several techniques are employed:

1. **Class-Aware Sampling**: During training, the model can be exposed to a balanced batch of images from both classes, either by oversampling the minority class or undersampling the majority class.

2. **Cost-Sensitive Learning**: Adjusting the loss function to penalize errors on the minority class more heavily can encourage the model to pay more attention to underrepresented classes.

3. **Data Augmentation**: Synthetic data generation, such as flipping, rotating, or cropping images of the minority class, can help balance the dataset.

4. **Class-Aware Pseudo-Labeling**: In semi-supervised learning, pseudo-labels can be generated for the minority class to augment the training data, improving the model’s performance on that class.

## Dataset

The datasets used in this project are **NI_6s** and **OW_6s** from IEEE Dataport, which are comprehensive and high-quality collections of satellite imagery specifically designed for the study of polar regions. These datasets are pivotal for advancing the understanding of sea ice and iceberg classification, offering a rich variety of imagery with different environmental conditions and ice formations.

### Dataset Description

- **NI_6s**: This dataset comprises satellite images focusing on Northern Hemisphere ice formations, providing detailed imagery that captures the variability and complexity of sea ice structures. The images are processed to a consistent resolution and are used to train and validate models that detect and classify ice formations in northern polar regions.

- **OW_6s**: This dataset includes satellite images primarily from the Southern Hemisphere, particularly around the Antarctic region. The dataset is critical for understanding iceberg formations and sea ice dynamics in this area. The imagery in OW_6s captures different types of sea ice, icebergs, and their interactions with the ocean environment.

### Data Characteristics

- **Resolution**: Both datasets provide high-resolution satellite images, making them suitable for fine-grained analysis and classification tasks. The images are stored in `.npy` format for efficient loading and processing.
- **Class Labels**: The datasets include binary labels corresponding to sea ice (`0`) and icebergs (`1`), enabling a clear distinction between the two classes.
- **Environmental Variability**: The datasets capture a range of environmental conditions, including different lighting, weather, and seasonal variations, which add complexity to the classification task and improve the model's robustness.

### Example Data Structure

- **NI_6s.npy**: Contains satellite images focused on Northern Hemisphere ice formations.
- **NI_6s_labels.npy**: Corresponding labels for the NI_6s images (0 for sea ice, 1 for iceberg).
- **OW_6s.npy**: Contains satellite images focused on Southern Hemisphere (Antarctic) ice formations.
- **OW_6s_labels.npy**: Corresponding labels for the OW_6s images (0 for sea ice, 1 for iceberg).

## Model Architecture

The CNN architecture used in this project is designed to balance complexity with efficiency, ensuring accurate classification without overfitting.

### Model Summary

The model consists of several convolutional layers for feature extraction, followed by max-pooling layers to downsample the feature maps. Finally, fully connected layers perform the binary classification based on the extracted features. Dropout layers are included to mitigate overfitting.

- **Input Shape**: (128, 128, 1) for grayscale images
- **Output**: Single neuron with sigmoid activation

 for binary classification

## Training and Evaluation

The model is trained using the Adam optimizer, which combines the advantages of Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). This optimizer adapts the learning rate for each parameter, enabling efficient and fast convergence.

### Training Parameters

- **Batch Size**: 32
- **Epochs**: 50
- **Validation Split**: 20%
- **Early Stopping**: Implemented to prevent overfitting, with a patience of 10 epochs.
- **Model Checkpointing**: Saves the best-performing model based on validation loss.

### Evaluation Metrics

The model is evaluated based on accuracy, precision, recall, and F1-score, with a particular focus on ensuring balanced performance across both classes.

### Example Output

#### Accuracy and Loss Curves

The training process produces accuracy and loss curves, which help visualize the model’s learning progression over time.

![1](https://github.com/user-attachments/assets/07248ede-470f-40b8-8bd1-2dcdbcacdcee)
*Figure 1: Training and Validation Accuracy over 50 epochs.*

![2](https://github.com/user-attachments/assets/a98d8a48-4206-4aa5-b469-efbe0a7c3417)
*Figure 2: Training and Validation Loss over 50 epochs.*

#### Confusion Matrix

The confusion matrix below shows the model's performance on the validation set, with a clear distinction between true positives, true negatives, false positives, and false negatives.
![3](https://github.com/user-attachments/assets/401eaf7d-51f4-4e39-b7dc-2e4114148014)  
*Figure 3: Confusion Matrix of the validation set.*

#### Classification Report

```plaintext
              precision    recall  f1-score   support

     sea_ice       0.98      0.97      0.98       500
    iceberg       0.97      0.98      0.97       500

    accuracy                           0.97      1000
   macro avg       0.97      0.97      0.97      1000
weighted avg       0.97      0.97      0.97      1000
```

## Conclusion

This project successfully demonstrates the application of deep learning techniques to the classification of sea ice and icebergs using satellite imagery. The use of CNNs allows for accurate, automated analysis, contributing to the broader field of climate science. By improving the accuracy of these classifications, the project helps enhance our understanding of polar regions and their role in the global climate system.

## Future Work

- **Extended Dataset**: Incorporate additional classes, such as different types of sea ice, to build a more comprehensive model.
- **Real-Time Deployment**: Implement the model in a real-time monitoring system, leveraging satellite data to provide continuous updates on polar conditions.
- **Transfer Learning**: Explore transfer learning from other large-scale image datasets to further improve model performance.

## References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv preprint arXiv:1804.02767.
- Johnson, J. M., & Khoshgoftaar, T. M. (2019). Survey on Deep Learning with Class Imbalance. Journal of Big Data, 6(1), 1-54.

## Acknowledgments

Thanks to the open-source community and contributors to TensorFlow and Keras, whose work made this project possible.
