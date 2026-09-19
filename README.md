# 🎓 Student Exam Performance Predictor

<p align="center">
  <b>Machine Learning Web Application for Predicting Student Mathematics Performance</b>
</p>

<p align="center">
  <a href="https://github.com/bharat-02/render-1-app">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge" alt="Machine Learning">
</p>

---

## 📌 About the Project

**Student Exam Performance Predictor** is a machine learning-powered web application that predicts a student's **Mathematics Score** based on academic and demographic information.

The application provides a simple web interface where users can enter student details such as:

* 👤 Gender
* 🌍 Race/Ethnicity
* 🎓 Parental Level of Education
* 🍽️ Lunch Type
* 📚 Test Preparation Course
* 📖 Reading Score
* ✍️ Writing Score

The submitted information is processed through a trained machine learning pipeline, and the application returns the predicted **Math Score**.

The Flask application exposes a home page and a prediction endpoint and runs on port `5000`.

---

## ✨ Features

* 🤖 Machine Learning based prediction
* 🌐 Flask web interface
* 📊 Student performance analysis
* 🔄 Reusable preprocessing and prediction pipeline
* 🧹 Data preprocessing and feature transformation
* 📈 Multiple ML algorithms supported during model experimentation
* 🚀 Production-ready Flask serving with Gunicorn support
* 📁 Modular project structure

---

## 🛠️ Tech Stack

| Technology      | Purpose                          |
| --------------- | -------------------------------- |
| 🐍 Python       | Core programming language        |
| 🌐 Flask        | Web application framework        |
| 🧠 Scikit-learn | Machine learning & preprocessing |
| 🐱 CatBoost     | Machine learning                 |
| 🚀 XGBoost      | Machine learning                 |
| 🐼 Pandas       | Data manipulation                |
| 🔢 NumPy        | Numerical computation            |
| 📊 Matplotlib   | Data visualization               |
| 📉 Seaborn      | Statistical visualization        |
| 💾 Dill         | Model/object serialization       |
| 🖥️ HTML/CSS    | Frontend interface               |
| ⚡ Gunicorn      | Production WSGI server           |

The repository's current `requirements.txt` includes Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, CatBoost, XGBoost, Flask, Dill, and Gunicorn.

---

## 🧠 Machine Learning Workflow

```text
                 ┌─────────────────────┐
                 │   Student Dataset   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Model Training      │
                 │ & Evaluation        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Trained ML Model    │
                 └──────────┬──────────┘
                            │
                            ▼
        ┌────────────────────────────────────┐
        │       Flask Prediction App         │
        └────────────────┬───────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ User Input          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Prediction Pipeline │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Predicted Math Score│
              └─────────────────────┘
```

---

## 📂 Project Structure

```text
render-1-app/
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── logs/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔄 Application Flow

### 1️⃣ User Input

The user enters student information through the web form.

### 2️⃣ Data Processing

The Flask application collects the submitted values and converts them into a structured DataFrame.

### 3️⃣ Prediction Pipeline

The application loads the trained preprocessing and machine learning objects and passes the input data through the prediction pipeline.

### 4️⃣ Prediction

The trained model generates a predicted Mathematics Score.

### 5️⃣ Result

The prediction is displayed on the web page with th
