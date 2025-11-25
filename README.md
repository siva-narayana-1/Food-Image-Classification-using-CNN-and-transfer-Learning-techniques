---

# 🍽️ Food Image Classification using CNN & Transfer Learning

This project is a deep learning–based **Food Image Classification** system built using **CNNs** and **Transfer Learning**. The model can identify different food categories from images and is suitable for restaurant automation, calorie estimation apps, menu recognition systems, and general computer vision applications.

---

## 📌 Features

* End-to-end food image classification pipeline
* Data preprocessing & augmentation
* Transfer Learning (VGG16 / ResNet50)
* High-accuracy model training
* Evaluation with accuracy, loss, and classification metrics
* Prediction script for new images
* Model saved in `.h5` format for deployment

---

## 📂 Project Structure

```
food-classification/
│── data/
│   ├── train/
│   ├── test/
│   └── validation/
│
│── notebooks/
│   └── food_classification.ipynb
│
│── model/
│   ├── food_model.h5
│   └── label_map.json
│
│── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/food-classification.git
cd food-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

The model uses **Transfer Learning** with frozen base layers and custom top layers.

* **Base Model:** VGG16 / ResNet50 (ImageNet pretrained)
* **Custom Layers:**

  * GlobalAveragePooling2D
  * Dense (ReLU)
  * Dropout
  * Dense (Softmax)

**Optimizer:** Adam
**Loss Function:** Categorical Crossentropy
**Metrics:** Accuracy

---

## 📊 Dataset Structure

You can use your own dataset or public datasets like Food-101.

Dataset folders should be structured as:

```
train/
   ├── biryani/
   ├── dosa/
   ├── idly/
   ├── pizza/
   ├── burger/
```

Each subfolder represents one class.

---

## 🏋️‍♂️ Training the Model

Run training using:

```bash
python src/train.py
```

The script includes:

* Image augmentation
* EarlyStopping
* ModelCheckpoint
* History plotting (accuracy/loss curves)

After training, model is saved to:

```
model/food_model.h5
model/label_map.json
```

---

## 🔍 Inference (Predicting Food From an Image)

To predict a food item from an image:

```bash
python src/predict.py --image path/to/image.jpg
```

Example Output:

```
Predicted Food: Biryani
Confidence: 98.4%
```

---

## 📈 Evaluation Metrics

The following metrics are generated:

* Accuracy
* Loss
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

Include graphs if available (accuracy & loss curves).

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib
* OpenCV
* Transfer Learning (VGG16 / ResNet50)

---

## 🚀 Future Enhancements

* Streamlit/FastAPI deployment
* TFLite conversion for mobile apps
* Integration with RAG for recipe generation
* Food calorie estimation model
* Data augmentation generator improvement

---
🌐 Live Demo (Render Deployment)

🔗 Live App:
https://food-image-classification-using-cnn-and-t0d7.onrender.com/

You can upload a food image and get instant predictions.

---

## 🙌 Author

**Siva Narayana Surya Chandra**
Machine Learning & Computer Vision Enthusiast

