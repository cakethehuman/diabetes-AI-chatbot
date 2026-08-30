from typing import Any

import joblib

import pandas as pd
import numpy as np

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
    
    def _predict_proba(self):
        logger.info("Making predictions")
        model = self.load_model()
        feature = ['smoking_history','gender','bmi','HbA1c_level','blood_glucose_level','age']
        df = pd.DataFrame([self.data], columns=feature)
        probability = model.predict_proba(df)
        return np.max(probability, axis=1) * 100, np.argmax(probability)
    
        
        
