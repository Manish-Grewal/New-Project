import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.utils import save_object,evaluate_models

@dataclass
class model_trainer_config:
    model_trainer_file_path=os.path.join('artifacts','model.pkl')
    
class model_trainer:
    def __init__(self):
        self.model_trainer_config=model_trainer_config()
        
    def initalize_model_trainer(test_array,train_array):
        try:
            logging.info("Splitting test and train data")
            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            logging.info("Initializing models")
            models={
                "LinearRegressor":LinearRegression(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "XGRegressor":XGBRegressor(),
                "KneighboursRegressor":KNeighborsRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "GradientBoostingRegressor":RandomForestRegressor()
            }
            
            params = {
                "LinearRegressor": {
                    "fit_intercept": [True, False]
                    },

                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 15, 20, 30],
                    "min_samples_split": [2, 5, 10, 15],
                    "min_samples_leaf": [1, 2, 4, 6],
                    "max_features": [None, "sqrt", "log2"]
                    },

                "XGRegressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "reg_alpha": [0, 0.1, 1],
                    "reg_lambda": [1, 5, 10]
                    },

                "KneighboursRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15, 20],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                    },
                
                "AdaBoostRegressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                    },

                "GradientBoostingRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 5, 7],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_features": [None, "sqrt", "log2"]
                    },

                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [1.0, "sqrt", "log2"],
                    "bootstrap": [True, False]
                    }
                }
            logging.info("Evaluating models")
        
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset")
            
            save_object(file_path=self.model_trainer_config.model_trainer_file_path,
                        obj=best_model)
            logging.info("Best model saved")
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
