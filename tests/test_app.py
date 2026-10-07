"""Small useful test suite for the Student Performance project.

Run from the project root with the venv activated::

    python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestArtifacts(unittest.TestCase):
    def test_model_and_preprocessor_load(self):
        from src.utils import load_object
        model = load_object(os.path.join("artifacts", "model.pkl"))
        pre = load_object(os.path.join("artifacts", "preprocessor.pkl"))
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(pre, "transform"))

    def test_prediction_valid_range(self):
        from src.pipeline.predict_pipeline import CustomData, PredictPipeline
        data = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=72,
            writing_score=74,
        )
        preds = PredictPipeline().predict(data.get_data_as_data_frame())
        self.assertEqual(len(preds), 1)
        self.assertFalse(bool(__import__("math").isnan(float(preds[0]))))
        self.assertGreaterEqual(max(0.0, min(100.0, float(preds[0]))), 0.0)
        self.assertLessEqual(max(0.0, min(100.0, float(preds[0]))), 100.0)

    def test_unseen_category_does_not_crash(self):
        from src.pipeline.predict_pipeline import CustomData, PredictPipeline
        data = CustomData(
            gender="female",
            race_ethnicity="group Z",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=70,
            writing_score=70,
        )
        preds = PredictPipeline().predict(data.get_data_as_data_frame())
        self.assertEqual(len(preds), 1)


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app
        flask_app.app.testing = True
        cls.client = flask_app.app.test_client()

    def test_index_route(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_predict_get(self):
        resp = self.client.get("/predictdata")
        self.assertEqual(resp.status_code, 200)

    def test_predict_post_valid(self):
        resp = self.client.post("/predictdata", data={
            "gender": "female",
            "ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "reading_score": "72",
            "writing_score": "74",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertRegex(resp.get_data(as_text=True), r"\d+\.\d{2}")


if __name__ == "__main__":
    unittest.main()
