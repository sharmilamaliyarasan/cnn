# 🔍 Handwritten Digit Recognizer Dashboard

## App Link:
https://dwirnjytrb6hvt23qbxjmj.streamlit.app/

## 📌 Project Overview

This project is an End-to-End Deep Learning application that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN).

It provides an interactive Streamlit dashboard where users can draw a digit or upload an image, get predictions in real-time, and visualize training performance.

## 📂 Data Preprocessing 
– Normalize images, reshape for CNN input, and split datasets into training/testing sets

⚙️ CNN Modeling – Multi-layer CNN architecture with tunable hyperparameters

📊 Evaluation Metrics – Accuracy, loss curves, confusion matrix

📈 Visualization – Training/validation loss and accuracy over epochs

## 🚀 Features

✅ CNN-based recognition of handwritten digits (0–9)

✅ Interactive Streamlit dashboard for live predictions via drawing or image upload

✅ Hyperparameter tuning (number of filters, kernel size, dropout rate, learning rate)

✅ Data normalization and reshaping for robust training

✅ Save/load trained CNN model (.h5) for fast predictions

✅ Visualization of training loss and accuracy curves

## 🛠️ Tech Stack

Python 🐍

TensorFlow / Keras → CNN modeling

NumPy / Pandas → Data handling

Matplotlib / Seaborn → Visualization

Joblib / h5py → Model persistence

OpenCV / Pillow → Image preprocessing

Streamlit → Interactive dashboard frontend

## 📂 Project Structure

Handwritten_Digit_Recognizer/

├── app.py

├── train.py

├── cnn_digit_model.h5

├── scaler.pkl (optional)

├── requirements.txt

└── README.md

## ⚙️ Installation & Setup

 Install dependencies:

pip install -r requirements.txt


 Run the Streamlit app:

streamlit run app.py

## 📊 Example Workflow

Preprocess dataset → Normalize, reshape, optionally augment images

Train CNN → Tune number of filters, kernel size, dropout, and layers

Evaluate model → Accuracy, loss curves, confusion matrix

Save trained CNN → .h5 file

Open Streamlit dashboard → Draw a digit or upload an image → Get prediction

## 📊 Evaluation Metrics

Accuracy → Baseline: ~98–99% on MNIST

Loss Curve → Training vs Validation loss per epoch

Confusion Matrix → Performance per digit class

## 🎯 Future Enhancements

Support multi-digit recognition (e.g., postal codes)

Export prediction reports (PDF/CSV)

Role-based access (Admin vs User)

Deploy on Hugging Face / Streamlit Cloud for public access

Integrate real-time camera input for digit recognition
