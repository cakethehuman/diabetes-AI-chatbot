from typing import Any, List, Optional

import joblib

import pandas as pd

from pydantic import BaseModel, model_validator

class Model(BaseModel):
    data: Optional[List[Any]] = None
    
    @model_validator(mode='after')
    def validate_data(self):
        if self.data is None:
            raise ValueError("Data is missing")
        return self
    
    def load_model(self):
        model = joblib.load('src/model/artifacts/trained_pipeline.joblib')
        return model
    
    def predict(self):
        model = self.load_model()
        feature = ['smoking_history','bmi','HbA1c_level','blood_glucose_level']
        df = pd.DataFrame([self.data], columns=feature)
        return model.predict(df)
    
e = Model(data=["never",0,0,0])
hasil_prediksi = e.predict()
print(hasil_prediksi)
        
        
