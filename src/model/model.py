from typing import Any

import joblib

import pandas as pd

from pydantic import BaseModel, model_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)

class Model(BaseModel):
    data: list[Any] | None = None
    
    @model_validator(mode='after')
    def validate_data(self):
        if self.data is None:
            raise ValueError("Data is missing")
        return self
    
    def load_model(self):
        model = joblib.load('src/model/artifacts/trained_pipeline.joblib')
        return model
    
    def predict(self):
        logger.info("Making predictions")
        model = self.load_model()
        feature = ['smoking_history','bmi','HbA1c_level','blood_glucose_level']
        df = pd.DataFrame([self.data], columns=feature)
        return model.predict(df)[0]
    
        
        
