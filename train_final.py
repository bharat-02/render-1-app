import os, sys
sys.path.insert(0, os.getcwd())
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Use existing split for fair before/after comparison.
    # Do NOT re-run ingestion (would rewrite train/test).
    dt = DataTransformation()
    train_arr, test_arr, pre_path = dt.initiate_data_transformation(
        "artifacts/train.csv", "artifacts/test.csv"
    )
    print("preprocessor saved to:", pre_path)
    print("train_arr:", train_arr.shape, "test_arr:", test_arr.shape)
    mt = ModelTrainer()
    r2 = mt.initiate_model_trainer(train_arr, test_arr)
    print("final test R2:", r2)
