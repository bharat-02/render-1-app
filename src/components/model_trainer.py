import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42, max_iter=20000),
                "ElasticNet": ElasticNet(random_state=42, max_iter=20000),
                "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_seed=42, allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    'max_depth':[None, 5, 10],
                },
                "Random Forest":{
                    'n_estimators': [200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                },
                "Extra Trees":{
                    'n_estimators': [200],
                    'max_depth': [None, 10],
                    'min_samples_split': [2, 5],
                },
                "Gradient Boosting":{
                    'learning_rate':[.05],
                    'subsample':[0.8],
                    'n_estimators': [200, 300],
                    'max_depth':[2, 3],
                },
                "Hist Gradient Boosting":{
                    'learning_rate':[.05, .1],
                    'max_iter':[200],
                    'max_depth':[None, 5],
                    'l2_regularization':[0.0, 1.0],
                },
                "Linear Regression":{},
                "Ridge":{
                    'alpha': [0.5, 1.0, 5.0],
                },
                "Lasso":{
                    'alpha': [0.005, 0.05],
                },
                "ElasticNet":{
                    'alpha': [0.005, 0.01],
                    'l1_ratio': [0.5, 0.8],
                },
                "XGBRegressor":{
                    'learning_rate':[.05],
                    'n_estimators': [200],
                    'max_depth': [3, 5],
                    'subsample': [0.8],
                },
                "CatBoosting Regressor":{
                    'depth': [4, 6],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [200],
                    'l2_leaf_reg': [1, 3],
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1, 0.5],
                    'n_estimators': [100, 200]
                }

            }

            model_report, model_details = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params)

            ## Selection rule: best CV R2 (never the test score) among models
            ## whose predictions are structurally confined to the observed
            ## target domain. Unregularized/regularized LINEAR models are
            ## unbounded affine functions: on below-support inputs (e.g.
            ## reading/writing scores under the training minima) they
            ## extrapolate outside 0-100 (measured: -6.42 at 0/0). They are
            ## therefore disqualified for this exam-score target, even though
            ## their in-distribution CV is marginally higher.
            unbounded = {"Linear Regression", "Ridge", "Lasso", "ElasticNet"}
            for name in unbounded:
                logging.info(
                    f"Disqualified {name} (CV R2={model_report[name]:.4f}): "
                    f"unbounded affine model, extrapolates outside 0-100 on "
                    f"below-support inputs)"
                )
            eligible = {n: s for n, s in model_report.items() if n not in unbounded}

            ## Best model is selected on CV R2 (not test R2) to avoid
            ## optimistic bias from test-set model selection.
            best_model_score = max(eligible.values())

            ## To get best model name from dict

            best_model_name = max(eligible, key=lambda n: eligible[n])
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            logging.info(f"Best model by CV R2: {best_model_name} ({best_model_score:.4f})")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            rmse = float((mean_squared_error(y_test, predicted)) ** 0.5)
            logging.info(
                f"Final {best_model_name}: test R2={r2_square:.4f} "
                f"MAE={mae:.4f} RMSE={rmse:.4f} "
                f"min={float(predicted.min()):.2f} max={float(predicted.max()):.2f}"
            )
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)