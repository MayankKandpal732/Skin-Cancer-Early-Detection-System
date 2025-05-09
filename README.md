# 🩺 **Skin Cancer Detection using Deep Learning** 🩺

## 🌟 **Project Overview**

In this project, I’ve developed a machine learning model to automatically classify skin cancer images into different categories — assisting in the early detection of skin cancer. Using **DenseNet121** with pre-trained weights, the model leverages state-of-the-art deep learning techniques to achieve high accuracy in distinguishing between cancerous and non-cancerous skin lesions.

### **Objective:**

* To develop a reliable and scalable deep learning model that can classify skin lesions into categories like **good**, **oil**, **scratch**, and **stain**, based on image data.
* Ultimately, the goal is to deploy the model in clinical settings to support dermatologists in making accurate and timely diagnoses.

---

## 🚀 **Key Features**

* **Deep Learning Architecture**: Built using **DenseNet121**, a powerful convolutional neural network (CNN) architecture pre-trained on ImageNet.
* **Fine-Tuning**: Initially frozen layers of DenseNet121 and later fine-tuned for better performance on the specific skin lesion dataset.
* **High Accuracy**: Achieved validation accuracy of over **71%** during training and fine-tuning stages.
* **Data Augmentation**: To generalize well on unseen images, various augmentation techniques (like rotation, zoom, and flipping) were applied to training images.
* **Early Stopping & Learning Rate Reduction**: Prevented overfitting and ensured smooth convergence with callbacks like **EarlyStopping** and **ReduceLROnPlateau**.

---

## 📂 **Project Structure**

```bash
skin_cancer_detection/
│
├── data/
│   ├── raw/              # Original dataset (good, oil, scratch, stain)
│   ├── masks/            # Ground truth for oil and scratch images
│
├── models/
│   ├── densenet121_skin_best.keras  # Saved model after training
│
├── notebooks/
│   ├── exploration.ipynb  # EDA and initial analysis
│
├── src/
│   ├── train_model.py     # Script for model training and evaluation
│
└── README.md             # This file
```

---

## 💻 **Installation & Setup**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/skin-cancer-detection.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧑‍💻 **How to Train the Model**

1. Download the dataset and extract it into the `data/` folder.
2. Run the following command to train the model:

   ```bash
   python src/train_model.py
   ```

This will start training the model with data augmentation, early stopping, and learning rate reduction, and the best model will be saved in the `models/` folder.

---

## 🧠 **Model Details**

* **Pre-trained Model**: **DenseNet121**, trained on ImageNet, used as the backbone.
* **Fine-Tuning**: Initially, only the classifier layer is trained, and later the entire network is fine-tuned with a very low learning rate.
* **Loss Function**: **Categorical Cross-Entropy**, as we are working with multiple categories.

---

## 📊 **Results**

* Achieved a **validation accuracy** of **\~79%** on the test set.
* The model was trained for 30 epochs, using a learning rate of **1e-4** with fine-tuning.
* 
---

## 🔧 **Technologies Used**

* **TensorFlow/Keras**: For building and training deep learning models.
* **NumPy** & **Pandas**: For data manipulation and analysis.
* **Matplotlib** & **Seaborn**: For visualizations.
* **OpenCV**: For image processing tasks.

---

## 📝 **Future Work**

* Experiment with **other architectures** like **EfficientNet**, **ResNet**, etc., for potentially better accuracy.
* Explore advanced techniques like **Mixup** and **CutMix** for further improving model robustness.
* Deploy the model as a web app for **real-time skin cancer detection**.

---

## 🤝 **Contributing**

Feel free to fork this project, open issues, and submit pull requests. Contributions are always welcome!

---

## 📧 **Contact**

If you have any questions or suggestions, feel free to contact me:

* Email: **[kandpalm04@gmail.com](mailto:kandpalm04@gmail.com)**
* GitHub: [Mayank Kandpal](https://github.com/yourusername)

---
