🎓 Student Exam Performance Predictor

<p align="center"> <b>Machine Learning Web Application for Predicting Student Mathematics Performance</b> </p>

<p align="center"> <a href="https://github.com/bharat-02/render-1-app"> <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" alt="GitHub"> </a> <a href="https://render-1-app.onrender.com"> <img src="https://img.shields.io/badge/Live-Demo-success?style=for-the-badge&logo=render" alt="Live Demo"> </a> <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"> <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" alt="Flask"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikit-learn" alt="Machine Learning"> </p>

🚀 Live Demo

🌐 Try the application online:

👉 https://render-1-app.onrender.com

The application allows users to enter student-related information and receive a predicted mathematics exam performance through a machine learning model.

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
| 💾 Dill         | Model/object serialization       |
| 🖥️ HTML/CSS    | Frontend interface               |
| ⚡ Gunicorn      | Production WSGI server           |

The repository's current `requirements.txt` includes Pandas, NumPy, Scikit-learn, CatBoost, XGBoost, Flask, Dill, and Gunicorn.

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
├── tests/
│   └── test_app.py
│
├── app.py
├── train_final.py
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

The prediction is displayed on the web page with the predicted Mathematics Score (clipped to 0–100).

---

## 🧪 ML Results (measured, Python 3.11 + scikit-learn 1.9.1)

- **Problem type:** regression. Target `math_score` (0–100). 1000 rows: 2 numeric (`reading_score`, `writing_score`) + 5 categoricals. No missing values, no duplicates.
- **Preprocessing:** `ColumnTransformer` — numeric median imputation + `StandardScaler`; categorical most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`. Fitted on train only (800/200 split, `random_state=42`).
- **Models tested (same split/preprocessing, 5-fold CV):** LinearRegression, Ridge, Lasso, ElasticNet, Huber, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, XGBoost, CatBoost, LightGBM, KNN, SVR.
- **Winner: ElasticNet (`alpha=0.005, l1_ratio=0.8`)** — best CV R² (0.8686); linear family beat all tree/boosting models on generalization with minimal training time.

| Metric | Original (LinearRegression) | Final (ElasticNet) |
|---|---:|---:|
| Train R² | 0.8743 | 0.8743 |
| CV R² | 0.8686 | 0.8686 |
| Test R² | 0.8804 | 0.8807 |
| MAE | 4.2148 | 4.2086 |
| RMSE | 5.3940 | 5.3889 |

- **GPU:** not used — 800×19 tabular data trains in seconds on CPU; GPU provides no benefit here.

---

## ▶️ Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_final.py   # retrain model + preprocessor into artifacts/
python app.py           # serves on http://127.0.0.1:5000
python -m unittest discover -s tests -v   # 6 tests
```

---

## 🚀 Deployment (Render)

- Production entry point: `gunicorn app:app` (`app.py` exposes both `application` and `app`).
- Local entry point `python app.py` serves on port 5000; production port is provided by Gunicorn/Render.
- `artifacts/model.pkl` + `artifacts/preprocessor.pkl` are committed so the app loads them at startup.
