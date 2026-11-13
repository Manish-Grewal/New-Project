import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.components.data_transgormation import Data_Transformation,DataTransformationConfig

@dataclass
class DataTransformationConfig:
    processor_path_obj_file_path: str = os.path.join("artifacts", "processor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        numerical_columns = ["reading_score", "writing_score"]
        categorical_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        logging.info("Numerical and categorical pipelines are created")

        return ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numerical_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ]
        )

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Train and test data are read")

            target_column = "math_score"
            train_features = train_data.drop(columns=[target_column])
            train_target = train_data[target_column]
            test_features = test_data.drop(columns=[target_column])
            test_target = test_data[target_column]
            logging.info("Train and test data are split into features and target")

            preprocessor = self.get_data_transformation_object()
            logging.info("Applying preprocessing to training and testing data")
            
            logging.info(f"Train features shape: {train_features.shape}")
            train_features_array = preprocessor.fit_transform(train_features)
            test_features_array = preprocessor.transform(test_features)

            logging.info(f"Train features array shape: {train_features_array.shape}")
            train_array = np.c_[train_features_array, train_target.to_numpy()]
            test_array = np.c_[test_features_array, test_target.to_numpy()]

            processor_path = (
                self.data_transformation_config.processor_path_obj_file_path
            )
            save_object(file_path=processor_path, obj=preprocessor)
            logging.info("Fitted preprocessing object is saved")

            return train_array, test_array, processor_path
        except Exception as e:
            raise CustomException(e, sys) from e


data_transformation = DataTransformation
data_transformationConfig = DataTransformationConfig
