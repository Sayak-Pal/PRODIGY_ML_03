---

# 🤖 Hand Gesture Recognition using Convolutional Neural Networks

This repository contains a deep learning project developed as part of the **Prodigy Infotech Internship**. The objective is to build a robust **Hand Gesture Recognition Model** that accurately classifies different hand gestures from **infrared image data** captured using the **Leap Motion sensor**.

---

## 📌 Project Description

The project involves classifying **10 distinct hand gestures** performed by **10 different individuals** using near-infrared images. These gestures are used to simulate **gesture-based human-computer interaction systems** that have real-world applications in **AR/VR**, **robotics**, **gaming**, and **sign language interpretation**.

This task is more complex than previous NLP or clustering tasks due to the **high dimensionality** and **spatial complexity** of image data. Developing this project greatly enhanced my understanding of:

* **Convolutional Neural Networks (CNNs)**
* **Image preprocessing and augmentation**
* **Model tuning and evaluation**

---

## 📂 Dataset

* **Source**: [LeapGestRecog Dataset on Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
* **Format**: Grayscale near-infrared gesture images
* **Classes**:

  * Palm
  * L
  * Fist
  * Fist\_moved
  * Thumb
  * Index
  * Ok
  * Palm\_moved
  * C
  * Down

---

## ⚙️ Technologies Used

* **Python 3.x**
* **TensorFlow & Keras** – Deep Learning (CNN model)
* **OpenCV** – Image processing
* **Pandas & NumPy** – Data handling
* **Matplotlib & Seaborn** – Visualization
* **scikit-learn** – Model evaluation metrics

---

## 🧠 Key Concepts Learned

* Image classification using **CNN architecture**
* Feature extraction from pixel grids
* **Data augmentation** and preprocessing using `ImageDataGenerator`
* Model validation, overfitting prevention, and confusion matrix analysis
* Understanding **why image recognition is harder** than text:

  > Text processing often deals with sequential, structured data, while images are high-dimensional, spatially complex, and require convolution-based operations for feature extraction.

---

## 🏗️ Model Architecture Summary

* Multiple **Conv2D** layers with BatchNormalization
* **MaxPooling2D** for dimensionality reduction
* **Dense layers** with ReLU activation and Dropout
* Final **Softmax** layer for multi-class classification

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog), extract it into the project directory.

4. Run the Python script or notebook:

   ```bash
   python hand_gesture_recognition.py
   ```

---

## 📈 Results

* Final Accuracy: **\~95%+** on test set
* Confusion matrix plotted for in-depth error analysis
* Trained model saved as `hand_gesture_Model.h5`

---

## 📁 File Structure

```
├── hand_gesture_recognition.py        # Main training & evaluation script
├── leapGestRecog/                     # Folder containing gesture image dataset
├── hand_gesture_Model.h5              # Saved trained model
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
```

---

## 📊 Visuals

* Model accuracy & loss plots
* Confusion matrix with class labels
* Sample prediction grid from test images

---

## 🏷️ Tags

`#DeepLearning` `#CNN` `#GestureRecognition` `#ComputerVision` `#TensorFlow` `#OpenCV` `#HumanComputerInteraction` `#ImageProcessing` `#Keras` `#ProdigyInfotech` `#InternshipProject` `#SignLanguageAI`

---
