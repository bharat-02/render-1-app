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

* 🤖 ML-based Maths Score prediction
* 🌐 Flask web interface
* 🎨 Pure HTML/CSS responsive frontend (no JavaScript, no UI frameworks)
* 🧹 Data preprocessing and feature transformation
* 📈 Multiple ML model evaluation with cross-validation
* 🏆 Final model selection by generalization performance
* 🔄 Reusable preprocessing and prediction pipeline
* ✅ HTML5 input validation (scores constrained to 0–100)
* 📊 CSS-only score visualization on the result card
* 🔒 Predicted score constrained to the valid 0–100 range
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
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Train/Validation
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Best Model
   ↓
Prediction Pipeline
   ↓
Flask Application
```

The request-handling flow in detail:

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
├── static/
│   └── css/
│       └── style.css
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
- **GPU:** not used — 800×19 tabular data trains in seconds on CPU; GPU provides no benefit here.

### Models evaluated

All models below were trained with the same split/preprocessing and tuned with
`GridSearchCV` (5-fold CV, selected on CV — never the test set). Boundary counts
are measured on the 200 test predictions.

| Model | Validation (CV R²) | Test R² | MAE | Min pred | Max pred | < 0 | > 100 |
|------|-------------------:|--------:|----:|---------:|---------:|----:|------:|
| CatBoost | 0.8590 | 0.8714 | 4.2755 | 17.93 | 92.21 | 0 | 0 |
| Gradient Boosting | 0.8537 | 0.8771 | 4.2469 | 13.61 | 93.96 | 0 | 0 |
| XGBoost | 0.8514 | 0.8679 | 4.3541 | 18.25 | 93.42 | 0 | 0 |
| Hist Gradient Boosting | 0.8452 | 0.8523 | 4.4682 | 27.15 | 94.39 | 0 | 0 |
| Random Forest | 0.8384 | 0.8558 | 4.5797 | 18.16 | 93.31 | 0 | 0 |
| Extra Trees | 0.8356 | 0.8505 | 4.6223 | 20.94 | 92.77 | 0 | 0 |
| AdaBoost | 0.8224 | 0.8475 | 4.7708 | 19.29 | 90.51 | 0 | 0 |
| Decision Tree | 0.7944 | 0.8242 | 4.9315 | 18.25 | 96.33 | 0 | 0 |

For reference (disqualified, see below): LinearRegression CV 0.8686 / test
0.8804, Ridge 0.8686/0.8806, Lasso 0.8686/0.8806, ElasticNet 0.8686/0.8807 —
marginally higher CV but structurally unbounded (raw −6.42 at reading/writing
0/0). A broader sweep (Huber, KNN, SVR, LightGBM) confirmed tree ensembles as
the only family that is both competitive and domain-valid.

### Root cause of the invalid predictions

The previous model (ElasticNet) is an **unbounded affine function**
(`prediction = intercept + coef · features`). On below-support inputs — e.g.
reading/writing scores under the training minima (24/15) — the standardized
features become large-negative z-scores (−4.5/−4.8 at 0/0) and the prediction
extrapolates outside the target domain (measured **−6.42 at 0/0**). This is
mathematics, not a data or pipeline bug: no leakage, wrong columns, or stale
artifacts were found. The fix is a model family whose predictions derive from
observed training targets (tree ensembles) — not app-level clipping.

### Best model

- **Selected model:** CatBoost (`depth=4`, `learning_rate=0.05`, `iterations=200`, `l2_leaf_reg=3`)
- **Why:** best CV R² (0.8590) among the naturally bounded tree-ensemble family,
  stable folds (±0.0098), lowest ensemble MAE (4.2755), and valid boundaries on
  test predictions and on extreme probe inputs (0/0 → 16.20).
- **Validation/CV:** R² 0.8590
- **Test:** R² 0.8714, MAE 4.2755, RMSE 5.5945
- **Boundaries:** test min 17.93 / max 92.21, 0 below 0, 0 above 100. One honest
  footnote: a single training prediction reaches 101.67 (boosting sums can
  marginally overshoot the training max); the below-zero failure mode is
  structurally eliminated everywhere (train min 14.68, all probes ≥ 16.20).

| Metric | Before (ElasticNet) | After (CatBoost) |
|---|---:|---:|
| Train R² | 0.8743 | 0.8883 |
| CV R² | 0.8686 | 0.8590 |
| Test R² | 0.8807 | 0.8714 |
| Test MAE | 4.2086 | 4.2755 |
| Test RMSE | 5.3889 | 5.5945 |
| Test invalid predictions | 0 (in-distribution) | 0 |
| Extreme-input (0/0) prediction | −6.42 (invalid) | 16.20 (valid) |

### Prediction range

The displayed Maths Score comes **directly from the model — `app.py` performs
no clipping or range modification** (only `f"{prediction:.2f}"` formatting).
Validity is a property of the selected model family: its predictions derive
from observed training targets, so unlike the previous affine model it does
not extrapolate below zero on low-score inputs.

---

## ▶️ Run locally

### Installation (Windows)

```bash
git clone https://github.com/bharat-02/render-1-app.git
cd render-1-app
```

Create and activate a virtual environment (Windows CMD):

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Start the app

```bash
python train_final.py   # retrain model + preprocessor into artifacts/
python app.py           # serves on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in a browser (the Flask console also shows the URL).

### Prediction usage

1. Open the prediction page (`Predict` in the nav bar, or `/predictdata`).
2. Select gender, race/ethnicity, parental education, lunch type, and test preparation course.
3. Enter the Reading Score (0–100).
4. Enter the Writing Score (0–100).
5. Click **Predict Maths Score**.
6. View the predicted Maths Score card (value out of 100 with a score bar).

### Testing

Tests live in `tests/` and use only the standard library (`unittest`, no extra
dependencies). They cover artifact loading, the prediction pipeline (valid
range, unseen categories), and the Flask routes including a form POST:

```bash
python -m unittest discover -s tests -v   # 6 tests
```

---

## 🚀 Deployment (Render)

- Production entry point: `gunicorn app:app` (`app.py` exposes both `application` and `app`).
- Local entry point `python app.py` serves on port 5000; production port is provided by Gunicorn/Render.
- `artifacts/model.pkl` + `artifacts/preprocessor.pkl` are committed so the app loads them at startup.
