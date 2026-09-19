from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


# =========================
# Home Page
# =========================

@app.route('/')
def index():
    return render_template('index.html')


# =========================
# Prediction Page
# =========================

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    # Open prediction page
    if request.method == 'GET':
        return render_template('home.html')

    else:

        # Get data from HTML form
        data = CustomData(
            gender=request.form.get('gender'),

            race_ethnicity=request.form.get('ethnicity'),

            parental_level_of_education=
                request.form.get('parental_level_of_education'),

            lunch=request.form.get('lunch'),

            test_preparation_course=
                request.form.get('test_preparation_course'),

            reading_score=
                float(request.form.get('reading_score')),

            writing_score=
                float(request.form.get('writing_score'))
        )

        # Convert input into DataFrame
        pred_df = data.get_data_as_data_frame()

        print("\nInput Data:")
        print(pred_df)

        print("Before Prediction")

        # Load prediction pipeline
        predict_pipeline = PredictPipeline()

        print("Mid Prediction")

        # Make prediction
        results = predict_pipeline.predict(pred_df)

        print("After Prediction")

        # Get predicted Maths Score
        prediction = float(results[0])

        # Keep prediction between 0 and 100
        prediction = max(0.0, min(100.0, prediction))

        print("Predicted Maths Score:", prediction)

        # Send prediction to HTML
        return render_template(
            'home.html',
            result=f"{prediction:.2f}"
        )


# =========================
# Run Flask Application
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)