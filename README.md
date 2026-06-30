# Heart Disease Prediction System using Artificial Intelligence & Machine Learning

A desktop-based **Heart Disease Prediction System** developed using **Python, Machine Learning, and Tkinter GUI**. The application predicts whether a patient is likely to have heart disease based on medical parameters and provides a risk percentage using a trained Random Forest Classifier.

---

## Project Overview

Heart disease is one of the leading causes of death worldwide. Early detection can help patients receive timely medical attention.

This project uses **Artificial Intelligence and Machine Learning** to analyze patient health parameters and predict the likelihood of heart disease.

The system provides an easy-to-use graphical interface where users can enter patient details and instantly receive prediction results.

---

## Features

- Machine Learning-based prediction
- Random Forest Classifier
- GUI developed using Tkinter
- Model Accuracy Display
- Heart Disease Detection
- Risk Percentage
- Risk Category (Low / Moderate / High)
- User-friendly Interface
- Clear Input Fields
- Exit Button

---

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Tkinter
- Matplotlib

---

## Project Structure

```
HeartDiseasePrediction/
│
├── heart.csv
├── heart_prediction.py
├── heart_gui.py
├── screenshots/
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses the **Heart Disease Dataset** containing **1025 patient records** and **14 medical attributes**.

### Dataset Features

- Age
- Gender
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Old Peak
- Slope
- CA
- Thal
- Target

---

## Machine Learning Algorithm

The prediction model is built using:

**Random Forest Classifier**

### Workflow

```
Dataset
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Random Forest Model
      ↓
Prediction
      ↓
GUI Output
```

---

## Model Accuracy

**Accuracy Achieved:** **98.54%**

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/HeartDiseasePrediction.git
```

Move into the project directory

```bash
cd HeartDiseasePrediction
```

Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

Run the application

```bash
python heart_gui.py
```

---

## 🖼 Screenshots

### Main Interface

![Main Interface](screenshots/home.png)

---

### Basic UI

![Basic UI](basicui.png)

---

### Healthy Prediction

![Healthy Prediction](screenshots/healthy.png)

---

### Risk

![Risk](screenshots/risk.png)
## Future Improvements

- Save prediction history
- Export reports as PDF
- Patient Login System
- Database Integration
- Deep Learning Models
- Web Version using Flask
- Cloud Deployment

---

## Developed By

**Aryan Pandey**

BCA + MCA (Dual Degree)

Amity Institute of Information Technology

---

## License

This project is developed for educational and internship purposes.
# heart_disease_prediction
