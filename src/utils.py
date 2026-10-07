import os
import sys

import numpy as np
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        details = {}

        for name in list(models.keys()):
            model = models[name]
            para = param[name]

            # CatBoost creates local working dirs; parallel GridSearch
            # workers collide on the same tmp dir, so keep it serial.
            n_jobs = 1 if "CatBoost" in name else 4

            gs = GridSearchCV(model, para, cv=5, n_jobs=n_jobs, scoring="r2")
            gs.fit(X_train, y_train)

            best = gs.best_estimator_

            y_train_pred = best.predict(X_train)
            y_test_pred = best.predict(X_test)
            cv_scores = cross_val_score(best, X_train, y_train, cv=5, scoring="r2", n_jobs=n_jobs)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

            # Select on CV mean (unbiased). Test metrics are reported only.
            report[name] = float(cv_scores.mean())
            details[name] = {
                "best_params": gs.best_params_,
                "train_r2": float(train_r2),
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
                "test_r2": float(test_r2),
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "estimator": best,
            }
            print(
                f"{name}: best={gs.best_params_} "
                f"train_R2={train_r2:.4f} "
                f"CV_R2={cv_scores.mean():.4f}+-{cv_scores.std():.4f} "
                f"test_R2={test_r2:.4f} MAE={test_mae:.4f} RMSE={test_rmse:.4f}"
            )

            # Keep the fitted best estimator for later saving
            models[name] = best

        return report, details

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)