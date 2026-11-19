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
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
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
