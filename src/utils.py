import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
import os,sys
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            param=params[list(models.keys())[i]]
            gs=GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)
            
            y_test_pred=gs.predict(X_test)
            
            test_model_score=r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
            
            
        return report
    except Exception as e:
        raise CustomException(e,sys)